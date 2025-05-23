# powerformer_encoder.py
# ──────────────────────
from __future__ import annotations
import liberate.fhe as fhe
from typing import Dict, List
import numpy as np
from pathlib import Path


def _interleave_heads(A, B):
    """(d,d) 두 행렬 → [a00,b00,a01,b01,…]  (row-major interleave)"""
    v1, v2 = A.reshape(-1), B.reshape(-1)
    out    = np.empty(v1.size + v2.size, dtype=np.float32)
    out[0::2], out[1::2] = v1, v2
    return out

def _encode_col_blocks(mat_head1, mat_head2, engine, level=0):

    num_rows_per_head = mat_head1.shape[0]
    num_cols = mat_head1.shape[1]
    blocks = []

    for j in range(num_cols):
        col_vec = np.empty(2 * num_rows_per_head, dtype=np.float32)
        col_vec[0::2] = mat_head1[:, j]
        col_vec[1::2] = mat_head2[:, j]
        
        try:
            encoded_block = engine.encode(col_vec, level=level, padding=False)
            blocks.append(encoded_block)

        except Exception as e_encode: # Keep basic error reporting for this critical function
            print(f"ERROR in _encode_col_blocks during engine.encode (col {j}, padding=False). Engine ID: {id(engine)}")
            print(f"Error details: {e_encode}")
            raise
            
    return blocks

def load_layer_weights(layer_dir: str, engine) -> dict:
    p = Path(layer_dir)
    WQ = np.load(p/'WQ.npy'); WK = np.load(p/'WK.npy'); WV = np.load(p/'WV.npy')
    WO = np.load(p/'WO.npy'); W1 = np.load(p/'W1.npy'); W2 = np.load(p/'W2.npy')

    WQ1, WQ2 = np.split(WQ, 2, axis=0)
    WK1, WK2 = np.split(WK, 2, axis=0)
    WV1, WV2 = np.split(WV, 2, axis=0)
    WO1, WO2 = np.split(WO, 2, axis=1)

    wq_blocks = _encode_col_blocks(WQ1, WQ2, engine) # level=0 is default
    wk_blocks = _encode_col_blocks(WK1, WK2, engine)
    wv_blocks = _encode_col_blocks(WV1, WV2, engine)
    
    wo_pt = engine.encode(_interleave_heads(WO1, WO2), level=0, padding=False)

    w1_pt = engine.encode(_interleave_heads(W1, W1),  level=0, padding=False)

    w2_pt = engine.encode(_interleave_heads(W2, W2),  level=0, padding=False)

    ln1_pt = engine.encode(np.load(p/'ln_attn.npy').reshape(-1), level=0) # padding 안씀 (기본값 사용)

    ln2_pt = engine.encode(np.load(p/'ln_ffn.npy' ).reshape(-1), level=0) # padding 안씀 (기본값 사용)
    
    weights = {
        "WQ": wq_blocks,
        "WK": wk_blocks,
        "WV": wv_blocks,
        "WO": wo_pt,
        "W1": w1_pt,
        "W2": w2_pt,
        "LN1": ln1_pt,
        "LN2": ln2_pt,
    }
    return weights

class PowerformerEncoder:
    """
    One encoder block of Powerformer-tiny.
    모든 가중치는 ckks_engine.encode() 로 미리 plaintext 객체(pt_)로 준비되어 있어야 한다.
    -----------------------------------------------------------------------
    Args
    ----
    engine  : ckks_engine           (cipher ops)
    evk     : evaluation-key bundle (rotate·relin 기 포함)
    params  : Dict[str,object]      (BRPmax/PolyReLU/LayerNorm 상수)
    weights : Dict[str,list]        (각 행렬에 대한 plaintext 리스트, 헤드 2개 → len==2)
        └─ "WQ","WK","WV","WO","W1","W2" : list[Plaintext]
        └─ "LN1","LN2"                     : γ,β → Plaintext 또는 부동소수
    """

    def __init__(self, engine, evk, params, w):
        self.engine, self.evk = engine, evk
        # ── 상수 ────────────────────────────────────────────────
        self.p_brpmax, self.c_brpmax, self.inv_Rd_brpmax = params["brpmax"].values()
        self.relu_c1_val = params["relu_poly"]["a1"]
        self.relu_c3_val = params["relu_poly"]["a3"]
        # ── 가중치 (이미 Plaintext) ────────────────────────────
        self.WQ_blocks = w["WQ"]      # list[Plaintext] 길이 128
        self.WK_blocks = w["WK"]
        self.WV_blocks = w["WV"]
        self.WO_pt     = w["WO"]      # Plaintext 한 장
        self.W1_pt     = w["W1"]
        self.W2_pt     = w["W2"]
        self.ln1_pt    = w["LN1"]     # γ,β interleave Plaintext
        self.ln2_pt    = w["LN2"]

        # Pre-encode plaintext constants
        # For _brpmax_poly
        self._pt_shift_c = self.engine.encode(
            np.full(2*128*128, self.c_brpmax, np.float32), level=0, padding=False
        )
        self._pt_invRd = self.engine.encode(
            np.full(2*128*128, self.inv_Rd_brpmax, np.float32), level=0, padding=False
        )

        # For _poly_relu
        self._pt_relu_a1 = self.engine.encode(
            np.full(2*128*128, self.relu_c1_val, np.float32), level=0, padding=False
        )
        self._pt_relu_a3 = self.engine.encode(
            np.full(2*128*128, self.relu_c3_val, np.float32), level=0, padding=False
        )

        # For _layernorm_poly
        d = 128
        slots_per_row = 2 * d
        self.inv_dim_val = 1.0 / d
        self._pt_inv_dim = self.engine.encode(
            np.full(slots_per_row * d, self.inv_dim_val, np.float32), level=0, padding=False
        )
        self.eps_val = 1e-5
        self._pt_eps = self.engine.encode(
            np.full(slots_per_row * d, self.eps_val, np.float32), level=0, padding=False
        )
        self.c0_val = 0.75
        self._pt_c0 = self.engine.encode(
            np.full(slots_per_row * d, self.c0_val, np.float32), level=0, padding=False
        )
        self.half_val = 0.5
        self.threehalf_val = 1.5
        self._pt_half = self.engine.encode(
            np.full(slots_per_row * d, self.half_val, np.float32), level=0, padding=False
        )
        self._pt_threehalf = self.engine.encode(
            np.full(slots_per_row * d, self.threehalf_val, np.float32), level=0, padding=False
        )   

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def __call__(self, ctx_in):
        """
        ctx_in : ciphertext – (X/X) packed 128×128 입력
        반환   : ciphertext – (다음 블록 입력 형) (X/X) packed
        """
        # ── 1. Multi-Head Attention (헤드 2개 병렬) ──────────────────
        ct_q = self._constrec_mm(ctx_in, self.WQ_blocks)               # Q
        ct_k = self._constrec_mm(ctx_in, self.WK_blocks)               # K
        ct_v = self._constrec_mm(ctx_in, self.WV_blocks)               # V

        ct_kT = self._transpose(ct_k)                           # Kᵀ
        ct_s  = self._block_mm(ct_q, ct_kT)                     # QKᵀ (이제 _block_mm_core 호출 후 추가 합산)

        ct_s  = self._brpmax_poly(ct_s)                         # BRPmax
        
        # Corrected MatMul(ct_s, ct_v) using the core block multiplication logic
        # ct_v is (V1/V2) packed, each V_i is 128x64.
        # ct_s is (S1/S2) packed, each S_i is 128x128.
        # We need Z_i = S_i * V_i. Result Z_i is 128x64.
        # _block_mm_core(A, B_T) computes A0*B0_T_block + A1*B1_T_block.
        # If A=ct_s and B_T=_transpose(ct_v), then this computes S*V.
        ct_z = self._block_mm_core(ct_s, self._transpose(ct_v)) # QKᵀ·V 

        ct_o  = self._const_mm(ct_z,  self.WO_pt)               # WO

        ct_out = self.engine.cc_add(ct_o, ctx_in)               # Add & Norm
        ct_out = self._layernorm_poly(ct_out, self.ln1_pt)

        # ── 2. Feed-Forward (poly-ReLU) ──────────────────────────
        ff1 = self._const_mm(ct_out, self.W1_pt)                # MatMul
        ff1 = self._poly_relu(ff1)                              # ReLU
        ff2 = self._const_mm(ff1, self.W2_pt)                   # MatMul

        ct_out = self.engine.cc_add(ff2, ct_out)                # Add & Norm
        ct_out = self._layernorm_poly(ct_out, self.ln2_pt)

        # ── 3. (선택) 부트스트랩 1 회 ────────────────────────────
        ct_out = self._bootstrap_if_needed(ct_out)

        return ct_out

    # ------------------------------------------------------------------ #
    # 내부 method
    # ------------------------------------------------------------------ #
    def _constrec_mm(self, ct_in, pt_blocks):
        # -----------------------------------------------------------
        #  _constrec_mm : (cipher X) × (plain W_blocks)  → (cipher Y)
        #                 ■ 입력  ct_in   : (X1/X2) packed 128×128
        #                 ■ 입력  pt_blocks[j] : Plaintext (2×128)  ←  W[:,j] interleave
        #                 ■ 출력  (Q1/Q2) packed 128×128
        # -----------------------------------------------------------
        engine = self.engine
        gk = self.evk.rotation_key

        d       = 128
        sqrt_d  = 11               # ⌈√128⌉  → Baby-step 크기
        accum   = None             # Giant-step 누적

        # ── Giant-step : 열 블록 단위 ----------------------------------------------
        for g in range(0, d, sqrt_d):        # g = 0,11,22,…121
            block_acc = None

            # ── Baby-step : 블록 내부 열마다 ----------------------------------------
            for k in range(sqrt_d):
                j = g + k
                if j >= d:
                    break

                # ① 입력 행렬 X 를 (2×j) 슬롯만큼 왼쪽 회전
                #    → 짝/홀 interleave 유지
                ct_rot = engine.rotate_galois(ct_in, gk, 2 * j)

                # ② 평문 열-벡터와 Hadamard 곱 (cipher × plain)
                prod   = engine.cm_mult(ct_rot, pt_blocks[j], evk=self.evk)
                engine.rescale(prod)

                # ③ 블록 누적
                block_acc = prod if block_acc is None else engine.cc_add(block_acc, prod)

            # ── Giant 누적
            accum = block_acc if accum is None else engine.cc_add(accum, block_acc)

        return accum          # (Q1/Q2) 짝/홀 패턴 그대로
    
    def _const_mm(self, ct_in, pt_W):
        """
        CONST algorithm  —  cipher (d×d)  ×  plain (d×d)  →  cipher
        • d = 128  →  회전 ≈ 4√d = 46  번
        • 짝 = 헤드-1,  홀 = 헤드-2  슬롯 구조 유지
        ----------------------------------------------------------------
        ct_in : ciphertext   (X1/X2), row-major interleave
        pt_W  : Plaintext    (W1/W2 interleave), row-major interleave
        """
        engine, gk = self.engine, self.evk.rotation_key
        d         = 128
        bs        = int(d ** 0.5)          # 11   (block size √d)

        # ── ❶ 마스크 블록 Plaintext 캐시 ----------------------------------
        cache_key = id(pt_W)               # 동일 가중치면 재사용
        if not hasattr(self, "_const_masks"):
            self._const_masks = {}
        if cache_key not in self._const_masks:
            masks = []                     # 길이 bs = 11
            decoded_pt_W_np = engine.decode(pt_W, level=0, is_real=False)

            for b_idx in range(bs):
                m = np.zeros(2 * d * d, dtype=np.float32)
                for col in range(b_idx, d, bs):            # bs-stride 열 선택
                    if (2*col+1) < len(decoded_pt_W_np):
                         m[2*col::2*d] = decoded_pt_W_np[2*col::2*d]      # 짝 슬롯
                         m[2*col+1::2*d] = decoded_pt_W_np[2*col+1::2*d]  # 홀 슬롯
                    elif (2*col) < len(decoded_pt_W_np):
                         m[2*col::2*d] = decoded_pt_W_np[2*col::2*d]

                masks.append(engine.encode(m, level=0, padding=False))
            self._const_masks[cache_key] = masks
        else:
            masks = self._const_masks[cache_key]

        # ── ❷ Giant-step over block index b --------------------------------
        accum = None
        for b_val, pt_mask in enumerate(masks):
            rot_step = 2 * b_val                         # 2·b_val
            ct_rot   = engine.rotate_galois(ct_in, gk, rot_step)

            prod     = engine.cm_mult(ct_rot, pt_mask, evk=self.evk)
            engine.rescale(prod)

            accum = prod if accum is None else engine.cc_add(accum, prod)

        return accum          # (Y1/Y2)  ‖  짝/홀 구조 그대로

    def _block_mm_core(self, ct_A, ct_B_T):
        """
        Core BLOCK algorithm part for A * B_T like multiplication.
        ct_A: Ciphertext for matrix A (e.g., Q or S). (d x 2k)
        ct_B_T: Ciphertext for matrix B_T (e.g., K_T or V_T). (2k x d')
        Result: A * B_T, specifically (A0*B0_T + A1*B1_T). (d x d')
        Input ciphertexts are (Head-1/Head-2) interleaved.
        Output ciphertext is also interleaved.
        """
        engine = self.engine 
        # gk = self.evk.rotation_key # Not needed for core multiplication, only for shifts

        # Split A into A0, A1 (column blocks)
        # Split B_T into B0_T, B1_T (row blocks of B_T)
        # This matches the Q = [Q0|Q1], K_T = [K0_T; K1_T] splitting logic.
        # rot_K_dim effectively defines the split point for the "k" dimension.
        # For QK_T: Q(d, k_dim_per_head), K_T(k_dim_per_head, d). k_dim_per_head=64.
        # So split occurs at 64th element of the "feature" dimension (2*64 slots).
        # For S*V_T: S(d,d), V_T(d_v_per_head, d). d_v_per_head=64.
        # If S is A, V_T is B_T.
        # A is split into A0, A1 (e.g., 128x64 column blocks if original A was 128x128).
        # B_T is split into B0_T, B1_T (e.g., 64x64 row blocks if original B_T was 64x128).
        # Then A0*B0_T is (128x64)*(64x64) -> (128x64). A1*B1_T is (128x64)*(64x64) -> (128x64).
        # Sum is (128x64). This is the desired shape for Z = S*V.
        
        # The rotation amount depends on the inner dimension 'k' of the block multiplication.
        # For QK^T, Q is d x k_h, K^T is k_h x d. rot_k_dim = 2 * k_h (k_h=64 for Q).
        # For S V^T, S is d x d, V^T is d_v x d. We need S_block * V_block_T.
        # S can be seen as [S_left | S_right] (d x d/2 each). V^T as [V_T_top ; V_T_bottom] (d_v/2 x d each).
        # The _block_mm logic splits based on 2*64 = 128 slots.
        # This corresponds to splitting a 128-wide feature map (interleaved heads) at midpoint.
        
        rot_split_dim = 2 * 64 # Corresponds to 64 features per head, total 128 slots for this dimension

        A0 = ct_A
        A1 = engine.rotate_galois(ct_A, self.evk.rotation_key, rot_split_dim) 

        B0_T = ct_B_T
        B1_T = engine.rotate_galois(ct_B_T, self.evk.rotation_key, -rot_split_dim) # Negative to bring the "second half" up

        prod0 = engine.cc_mult(A0, B0_T, evk=self.evk)
        engine.rescale(prod0)
        prod1 = engine.cc_mult(A1, B1_T, evk=self.evk)
        engine.rescale(prod1)

        # Result is A0*B0_T + A1*B1_T
        Result_AB_T = engine.cc_add(prod0, prod1)
        return Result_AB_T

    def _block_mm(self, ct_q, ct_kT):
        """
        BLOCK algorithm  —  cipher Q(128×64/head) × cipher Kᵀ(64×128/head) → cipher Score (128x128/head)
        • 입력 두 암호문 모두 (짝=Head-1, 홀=Head-2) interleave
        • 회전표 stride = 2×64 = 128  /  2×8 = 16  등 6개만 필요
        • 반환 암호문도 interleave 유지
        This function now computes QK_T specifically, including the final row accumulation.
        """
        engine, gk = self.engine, self.evk.rotation_key

        # Core multiplication QK_T_core = Q0*K0_T + Q1*K1_T
        S_core = self._block_mm_core(ct_q, ct_kT)

        # ── 3. 행 누적(세로 reduce)   —  회전 32,16,8,4,2  ─────────────
        # This part is specific to QK^T for score calculation.
        S_accum = S_core
        for shift in [32,16,8,4,2]: 
            rot = engine.rotate_galois(S_accum, gk, shift*2)     # 짝수 stride
            S_accum = engine.cc_add(S_accum, rot)
        
        return S_accum

    def _transpose(self, ct_in):
        """
        Row-major (X/X) ciphertext → (X/X) 전치
        - 입력/출력 모두 짝-홀 interleave 유지
        - 회전 stride = 2*(d-1) = 254  한 종류
        """
        engine, gk = self.engine, self.evk.rotation_key
        d       = 128
        stride  = 2 * (d - 1)          # 254  (짝/홀 안전)

        # ── 대각선 마스크 Plaintext를 (lazy) 캐시 ──────────────────
        if not hasattr(self, "_transpose_masks"):
            masks = []
            for k_val in range(d):
                # M_k: 1 at positions where (c - r) mod d == k
                m = np.zeros(2 * d * d, dtype=np.float32)
                for r in range(d):
                    c = (r + k_val) % d
                    idx_even = 2 * (r * d + c)     # Head-1 슬롯
                    idx_odd  = idx_even + 1        # Head-2 슬롯
                    m[idx_even] = 1.0
                    m[idx_odd]  = 1.0
                masks.append(engine.encode(m, level=0, padding=False))
            self._transpose_masks = masks

        # ── 전치 계산 -------------------------------------------------
        acc = None
        for k_idx in range(d):
            rot = engine.rotate_galois(ct_in, gk, stride * k_idx)   # 254·k_idx  회전
            prod = engine.cm_mult(rot, self._transpose_masks[k_idx], evk=self.evk)
            engine.rescale(prod)
            acc  = prod if acc is None else engine.cc_add(acc, prod)

        return acc          # (Kᵀ₁ / Kᵀ₂)  interleave 유지

    def _brpmax_poly(self, ct):
        """
        ReLU(x+c)^p / R_d     (보통 p=2,3)
        """
        engine = self.engine

        # ① shift : x ← x + c
        ct_shift = engine.cc_add(ct, self._pt_shift_c)

        # ② Poly-ReLU
        ct_relu  = self._poly_relu(ct_shift)

        # ③ 거듭제곱  (p 차례 cc_mult · rescale)
        if self.p_brpmax == 2:
            ct_pow = engine.cc_mult(ct_relu, ct_relu, evk=self.evk)
            engine.rescale(ct_pow)
        elif self.p_brpmax == 3:
            sq = engine.cc_mult(ct_relu, ct_relu, evk=self.evk); engine.rescale(sq)
            ct_pow = engine.cc_mult(sq, ct_relu, evk=self.evk);  engine.rescale(ct_pow)
        else:
            raise ValueError("Only p=2 or 3 supported for BRPmax")

        # ④ 나누기 R_d  (평문 상수 곱)
        ct_out = engine.cm_mult(ct_pow, self._pt_invRd, evk=self.evk)
        engine.rescale(ct_out)

        return ct_out     # 여전히 짝/홀 헤드 분리 유지

    def _poly_relu(self, ct):
        """
        a1·x + a3·x³   (짝/홀 interleave 유지)
        Using a1 = 0.5, a3 = 0.125 as per common approximations,
        but using self._pt_relu_a1 and self._pt_relu_a3 which are pre-encoded.
        """
        engine = self.engine
        # x³
        ct_sq = engine.cc_mult(ct, ct, evk=self.evk)
        engine.rescale(ct_sq)
        ct_cu = engine.cc_mult(ct_sq, ct, evk=self.evk)
        engine.rescale(ct_cu)

        # a1·x   ,   a3·x³
        term1 = engine.cm_mult(ct, self._pt_relu_a1, evk=self.evk)
        engine.rescale(term1)
        term2 = engine.cm_mult(ct_cu, self._pt_relu_a3, evk=self.evk)
        engine.rescale(term2)

        return engine.cc_add(term1, term2)

    def _layernorm_poly(self, ct, ln_pt):
        """
        LayerNorm(x) ≈ γ · (x−μ) · inv_sqrt(var+ε) + β
        ------------------------------------------------------------------
        ct      : ciphertext  — (X1/X2) 한 행렬
        ln_pt   : Plaintext   — [γ_even, β_even, γ_odd, β_odd, …] interleave
                (encode 시 Head-1 슬롯에 γ,β / Head-2 슬롯에 γ,β)
        ------------------------------------------------------------------
        CKKS 제약상
        - 행(row) 단위 평균·분산 : flatten 256-slot (= 128·2) 블록마다
        - Newton 2-스텝 :  y ← y·(1.5 − 0.5·v·y²)
        """
        engine = self.engine
        gk  = self.evk.rotation_key
        d   = 128
        slots_per_row = 2 * d

        # ── 1. Row-wise SUM (tree reduce) ------------------------------
        def row_sum(ct_in_local):
            """256-slot 구간 합을 모든 슬롯에 broadcast"""
            ct_out_local = ct_in_local
            for shift in [1,2,4,8,16,32,64,128]:
                rot = engine.rotate_galois(ct_out_local, gk, shift)
                ct_out_local = engine.cc_add(ct_out_local, rot)
            return ct_out_local

        ct_sum = row_sum(ct)                 # Σ x_i  (broadcast)
        mu = engine.cm_mult(ct_sum, self._pt_inv_dim, evk=self.evk)
        engine.rescale(mu)

        # ── 2. Centering ----------------------------------------------
        ct_center = engine.cc_sub(ct, mu)

        # ── 3. Variance ------------------------------------------------
        ct_sq   = engine.cc_mult(ct_center, ct_center, evk=self.evk)
        engine.rescale(ct_sq)
        var_sum = row_sum(ct_sq)
        var     = engine.cm_mult(var_sum, self._pt_inv_dim, evk=self.evk)    # E[(x-μ)²]
        engine.rescale(var)

        # ── 4. inv_sqrt via Newton (2 iter) ----------------------------
        v_eps = engine.cc_add(var, self._pt_eps)

        y = engine.cm_mult(v_eps, self._pt_c0, evk=self.evk)     # rough 1/√v
        engine.rescale(y)

        # 1st Newton
        y_sq = engine.cc_mult(y, y, evk=self.evk); engine.rescale(y_sq)
        term = engine.cc_mult(v_eps, y_sq, evk=self.evk); engine.rescale(term)
        term = engine.cm_mult(term, self._pt_half, evk=self.evk); engine.rescale(term)
        y    = engine.cc_mult(y, engine.cc_sub(self._pt_threehalf, term), evk=self.evk); engine.rescale(y)

        # 2nd Newton
        y_sq = engine.cc_mult(y, y, evk=self.evk); engine.rescale(y_sq)
        term = engine.cc_mult(v_eps, y_sq, evk=self.evk); engine.rescale(term)
        term = engine.cm_mult(term, self._pt_half, evk=self.evk); engine.rescale(term)
        y    = engine.cc_mult(y, engine.cc_sub(self._pt_threehalf, term), evk=self.evk); engine.rescale(y)

        # ── 5. Normalize ----------------------------------------------
        x_hat = engine.cc_mult(ct_center, y, evk=self.evk); engine.rescale(x_hat)

        # ── 6. Scale(γ) & Shift(β) ------------------------------------
        x_scaled = engine.cm_mult(x_hat, ln_pt, evk=self.evk)
        engine.rescale(x_scaled)
        out   = engine.cc_add(x_scaled, ln_pt)      # +β

        return out

    def _bootstrap_if_needed(self, ct):
        """층당 1 회 부트스트랩 (Liberate 0.x는 rescale+level_up 사용)"""
        # TODO: check level, then call engine.level_up(ct, target_level) or a bootstrap function
        return ct