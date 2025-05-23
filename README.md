# Liberate 

## Project Structure
```
powerformer-liberate-main/
├── README.md                   # Project documentation
├── main.py                     # Main execution script
├── client.py                   # Client-side implementation for FHE
├── server.py                   # Server-side implementation for FHE
├── powerformer_encoder.py      # Core Powerformer encoder implementation
├── ckks_engine.py             # CKKS homomorphic encryption engine
├── knowledge_distillation.py   # Knowledge distillation implementation
├── bert_base_train.py         # BERT base model training script
├── student_powerformer_rte/    # Student model directory
│   ├── powerformer_encoder_config.json
│   ├── layer00/
│   │   └── [weight files]
│   ├── layer01/
│   │   └── [weight files]
│   └── ...
├── student_powerformer_rte_2/  # Alternative student model directory
│   └── [similar structure as above]
├── bert-base-finetuned-rte/    # Fine-tuned BERT base model
│   └── [model files]
└── __pycache__/               # Python cache directory
```

## ckks_engine Class Methods

| Method                                     | Parameters                                                                                                 | Description                                                                                                                               | Description-KR |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__`                                 | `devices: list[int] = None`, `verbose: bool = False`, `bias_guard: bool = True`, `norm: str = 'forward'`, `**ctx_params` | Initializes the `ckks_engine` object with context parameters and sets up NTT, RNG, and other necessary pre-calculations.                  | `ckks_engine` 객체를 초기화하고 컨텍스트 매개변수를 설정하며 NTT, RNG 및 기타 필요한 사전 계산을 설정한다. |
| `create_rescale_scales`                    | `self`                                                                                                     | Creates scales used for rescaling ciphertexts after multiplication.                                                                     | 곱셈 연산 후 암호문 재조정에 사용되는 스케일을 생성한다. |
| `leveled_devices`                          | `self`                                                                                                     | Determines the number of devices and neighbor devices for each level.                                                                     | 각 레벨에 대한 디바이스 수와 이웃 디바이스를 결정한다. |
| `alloc_parts`                              | `self`                                                                                                     | Allocates parts for RNS components and determines storage IDs.                                                                          | RNS 구성 요소에 대한 부분을 할당하고 저장소 ID를 결정힌다. |
| `create_ksk_rescales`                      | `self`                                                                                                     | Creates rescaling factors for key switching keys (KSK).                                                                                   | 키 전환 키(KSK)에 대한 재조정 인자를 생성한다. |
| `reserve_ksk_buffers`                      | `self`                                                                                                     | Reserves memory buffers for KSK operations.                                                                                             | KSK 연산을 위한 메모리 버퍼를 예약한다. |
| `make_mont_PR`                             | `self`                                                                                                     | Precomputes Montgomery representation of P*R for KSK generation.                                                                        | KSK 생성을 위한 P*R의 몽고메리 표현을 사전 계산한다. |
| `make_adjustments_and_corrections`         | `self`                                                                                                     | Calculates scaling factor adjustments and corrections for decoding.                                                                     | 디코딩을 위한 스케일 인자 조정과 보정을 계산한다. |
| `absmax_error`                             | `self`, `x`, `y`                                                                                           | Calculates the absolute maximum error between two arrays `x` and `y`.                                                                     | 두 배열 `x`와 `y` 사이의 절대 최대 오차를 계산한다. |
| `integral_bits_available`                  | `self`                                                                                                     | Calculates the number of available bits for the integral part of a plaintext.                                                           | 평문의 정수 부분에 사용 가능한 비트 수를 계산한다. |
| `example`                                  | `self`, `amin=None`, `amax=None`, `decimal_places: int = 10`                                              | Generates an example complex-valued plaintext vector for testing.                                                                         | 테스트용 복소수 평문 벡터를 생성한다. |
| `padding`                                  | `self`, `m`                                                                                                | Pads a message `m` to the required number of slots.                                                                                       | 메시지 `m`을 필요한 슬롯 수에 맞게 패딩한다. |
| `encode`                                   | `self`, `m`, `level: int = 0`, `padding=True`                                                              | Encodes a plaintext message `m` into the RNS polynomial representation.                                                                   | 평문 메시지 `m`을 RNS 다항식 표현으로 인코딩한다. |
| `decode`                                   | `self`, `m`, `level=0`, `is_real: bool = False`                                                            | Decodes an RNS polynomial `m` back to a plaintext message.                                                                                | RNS 다항식 `m`을 평문 메시지로 디코딩한다. |
| `create_secret_key`                        | `self`, `include_special: bool = True`                                                                     | Creates a secret key for the CKKS scheme.                                                                                                 | CKKS 스키마를 위한 비밀키를 생성한다. |
| `create_public_key`                        | `self`, `sk: data_struct`, `include_special: bool = False`, `a: list[torch.Tensor] = None`                  | Creates a public key corresponding to the given secret key `sk`.                                                                          | 주어진 비밀키 `sk`에 대응하는 공개키를 생성한다. |
| `encrypt`                                  | `self`, `pt: list[torch.Tensor]`, `pk: data_struct`, `level: int = 0`                                      | Encrypts a plaintext `pt` using the public key `pk`.                                                                                      | 공개키 `pk`를 사용하여 평문 `pt`를 암호화한다. |
| `decrypt_triplet`                          | `self`, `ct_mult: data_struct`, `sk: data_struct`, `final_round=True`                                     | Decrypts a ciphertext triplet `ct_mult` (resulting from multiplication before relinearization) using the secret key `sk`.                 | 비밀키 `sk`를 사용하여 암호문 삼중항 `ct_mult`(재선형화 전의 곱셈 결과)를 복호화한다. |
| `decrypt_double`                           | `self`, `ct: data_struct`, `sk: data_struct`, `final_round=True`                                          | Decrypts a standard ciphertext doublet `ct` using the secret key `sk`.                                                                  | 비밀키 `sk`를 사용하여 표준 암호문 이중항 `ct`를 복호화한다. |
| `decrypt`                                  | `self`, `ct: data_struct`, `sk: data_struct`, `final_round=True`                                          | Decrypts a ciphertext (either doublet or triplet) using the secret key `sk`.                                                              | 비밀키 `sk`를 사용하여 암호문(이중항 또는 삼중항)을 복호화한다. |
| `create_key_switching_key`                 | `self`, `sk_from: data_struct`, `sk_to: data_struct`, `a=None`                                             | Creates a key switching key to transform a ciphertext encrypted under `sk_from` to one encrypted under `sk_to`.                           | `sk_from`으로 암호화된 암호문을 `sk_to`로 암호화된 형태로 변환하기 위한 키 전환 키를 생성한다. |
| `pre_extend`                               | `self`, `a`, `device_id`, `level`, `part_id`, `exit_ntt=False`                                            | Performs the first phase of basis extension for key switching.                                                                          | 키 전환을 위한 기저 확장의 첫 번째 단계를 수행한다. |
| `extend`                                   | `self`, `state`, `device_id`, `level`, `part_id`, `target_device_id=None`                                 | Performs the second phase of basis extension for key switching.                                                                         | 키 전환을 위한 기저 확장의 두 번째 단계를 수행한다. |
| `create_switcher`                          | `self`, `a: list[torch.Tensor]`, `ksk: data_struct`, `level`, `exit_ntt=False`                             | Creates the switching polynomials for key switching based on the input polynomial `a` and KSK.                                        | 입력 다항식 `a`와 KSK를 기반으로 키 전환을 위한 전환 다항식을 생성한다. |
| `switcher_later_part`                      | `self`, `state`, `ksk`, `src_device_id`, `dst_device_id`, `level`, `part_id`                                | Performs the later part of the key switching operation on a specific device.                                                              | 특정 디바이스에서 키 전환 연산의 후반부를 수행한다. |
| `switch_key`                               | `self`, `ct: data_struct`, `ksk: data_struct`                                                              | Switches the key of a ciphertext `ct` using the key switching key `ksk`.                                                                  | 키 전환 키 `ksk`를 사용하여 암호문 `ct`의 키를 전환한다. |
| `rescale`                                  | `self`, `ct: data_struct`, `exact_rounding=True`                                                           | Rescales a ciphertext `ct` to reduce its modulus and manage noise growth.                                                                 | 암호문 `ct`의 모듈러스를 줄이고 노이즈 증가를 관리하기 위해 재조정한다. |
| `create_evk`                               | `self`, `sk: data_struct`                                                                                  | Creates an evaluation key (relinearization key) from a secret key `sk`.                                                                   | 비밀키 `sk`로부터 평가 키(재선형화 키)를 생성한다. |
| `cc_mult`                                  | `self`, `a: data_struct`, `b: data_struct`, `evk: data_struct`, `relin=True`                               | Performs ciphertext-ciphertext multiplication of `a` and `b`, optionally relinearizing the result using `evk`.                            | 암호문 `a`와 `b`의 곱셈을 수행하고, 선택적으로 `evk`를 사용하여 결과를 재선형화한다. |
| `relinearize`                              | `self`, `ct_triplet: data_struct`, `evk: data_struct`                                                      | Relinearizes a ciphertext triplet (from multiplication) to a doublet using the evaluation key `evk`.                                    | 평가 키 `evk`를 사용하여 암호문 삼중항(곱셈 결과)을 이중항으로 재선형화한다. |
| `create_rotation_key`                      | `self`, `sk: data_struct`, `delta: int`, `a: list[torch.Tensor] = None`                                    | Creates a rotation key for rotating a ciphertext by `delta` slots, using secret key `sk`.                                                 | 비밀키 `sk`를 사용하여 암호문을 `delta` 슬롯만큼 회전시키기 위한 회전 키를 생성한다. |
| `rotate_single`                            | `self`, `ct: data_struct`, `rotk: data_struct`                                                             | Rotates a ciphertext `ct` using a single rotation key `rotk`.                                                                           | 단일 회전 키 `rotk`를 사용하여 암호문 `ct`를 회전시킨다. |
| `create_galois_key`                        | `self`, `sk: data_struct`                                                                                  | Creates a set of Galois keys for all powers-of-two rotations from a secret key `sk`.                                                      | 비밀키 `sk`로부터 모든 2의 거듭제곱 회전을 위한 갈루아 키 세트를 생성한다. |
| `rotate_galois`                            | `self`, `ct: data_struct`, `gk: data_struct`, `delta: int`, `return_circuit=False`                         | Rotates a ciphertext `ct` by `delta` slots using Galois keys `gk`. Optionally returns the sequence of rotations performed.                | 갈루아 키 `gk`를 사용하여 암호문 `ct`를 `delta` 슬롯만큼 회전시킵니다. 선택적으로 수행된 회전 순서를 반환한다. |
| `cc_add_double`                            | `self`, `a: data_struct`, `b: data_struct`                                                                 | Performs ciphertext-ciphertext addition for two standard ciphertext doublets `a` and `b`.                                                 | 두 표준 암호문 이중항 `a`와 `b`의 암호문-암호문 덧셈을 수행한다. |
| `cc_add_triplet`                           | `self`, `a: data_struct`, `b: data_struct`                                                                 | Performs ciphertext-ciphertext addition for two ciphertext triplets `a` and `b`.                                                          | 두 암호문 삼중항 `a`와 `b`의 암호문-암호문 덧셈을 수행한다. |
| `cc_add`                                   | `self`, `a: data_struct`, `b: data_struct`                                                                 | Performs ciphertext-ciphertext addition, dispatching to `cc_add_double` or `cc_add_triplet` based on input types.                        | 입력 유형에 따라 `cc_add_double` 또는 `cc_add_triplet`으로 암호문-암호문 덧셈을 수행한다. |
| `cc_sub_double`                            | `self`, `a: data_struct`, `b: data_struct`                                                                 | Performs ciphertext-ciphertext subtraction for two standard ciphertext doublets `a` and `b`.                                                | 두 표준 암호문 이중항 `a`와 `b`의 암호문-암호문 뺄셈을 수행한다. |
| `cc_sub_triplet`                           | `self`, `a: data_struct`, `b: data_struct`                                                                 | Performs ciphertext-ciphertext subtraction for two ciphertext triplets `a` and `b`.                                                         | 두 암호문 삼중항 `a`와 `b`의 암호문-암호문 뺄셈을 수행한다. |
| `cc_subtract`                              | `self`, `a`, `b`                                                                                           | Alias for `cc_sub`.                                                                                                                       | `cc_sub`의 별칭. |
| `level_up`                                 | `self`, `ct: data_struct`, `dst_level: int`                                                                | Adjusts the level of a ciphertext `ct` to a target `dst_level`.                                                                           | 암호문 `ct`의 레벨을 목표 레벨 `dst_level`로 조정한다. |
| `encodecrypt`                              | `self`, `m`, `pk: data_struct`, `level: int = 0`, `padding=True`                                           | Fused operation of encoding a plaintext `m` and then encrypting it with public key `pk`.                                                  | 평문 `m`을 인코딩한 후 공개키 `pk`로 암호화하는 융합 연산을 수행한다. |
| `decryptcode`                              | `self`, `ct: data_struct`, `sk: data_struct`, `is_real=False`, `final_round=True`                          | Fused operation of decrypting a ciphertext `ct` with secret key `sk` and then decoding it.                                                | 비밀키 `sk`로 암호문 `ct`를 복호화한 후 디코딩하는 융합 연산을 수행한다. |
| `encorypt`                                 | `self`, `m`, `pk: data_struct`, `level: int = 0`, `padding=True`                                           | Alias for `encodecrypt`.                                                                                                                  | `encodecrypt`의 별칭. |
| `decrode`                                  | `self`, `ct: data_struct`, `sk: data_struct`, `is_real=False`, `final_round=True`                          | Alias for `decryptcode`.                                                                                                                  | `decryptcode`의 별칭. |
| `create_conjugation_key`                   | `self`, `sk: data_struct`                                                                                  | Creates a key for performing complex conjugation on a ciphertext, using secret key `sk`.                                                  | 비밀키 `sk`를 사용하여 암호문에 대한 복소수 켤레 연산을 수행하기 위한 키를 생성한다. |
| `conjugate`                                | `self`, `ct: data_struct`, `conjk: data_struct`                                                            | Performs complex conjugation on a ciphertext `ct` using the conjugation key `conjk`.                                                      | 켤레 키 `conjk`를 사용하여 암호문 `ct`에 대한 복소수 켤레 연산을 수행한다. |
| `clone_tensors`                            | `self`, `data: data_struct`                                                                                | Creates a deep copy of the tensor data within a `data_struct`.                                                                          | `data_struct` 내의 텐서 데이터의 깊은 복사본을 생성한다. |
| `clone`                                    | `self`, `text`                                                                                             | Creates a deep copy of a `data_struct` object, including its tensor data.                                                               | 텐서 데이터를 포함한 `data_struct` 객체의 깊은 복사본을 생성한다. |
| `download_to_cpu`                          | `self`, `gpu_data`, `level`, `include_special`                                                             | Moves tensor data from GPU(s) to the CPU.                                                                                                 | GPU에서 CPU로 텐서 데이터를 이동한다. |
| `upload_to_gpu`                            | `self`, `cpu_data`, `level`, `include_special`                                                             | Moves tensor data from the CPU to GPU(s).                                                                                                 | CPU에서 GPU로 텐서 데이터를 이동한다. |
| `move_tensors`                             | `self`, `data`, `level`, `include_special`, `direction`                                                    | Moves tensor data between CPU and GPU based on the specified `direction`.                                                                 | 지정된 `direction`에 따라 CPU와 GPU 간에 텐서 데이터를 이동한다. |
| `move_to`                                  | `self`, `text`, `direction='gpu2cpu'`                                                                      | Moves an entire `data_struct` object (and its contained tensors) between CPU and GPU.                                                     | 전체 `data_struct` 객체(및 포함된 텐서)를 CPU와 GPU 간에 이동한다. |
| `cpu`                                      | `self`, `ct`                                                                                               | Shortcut for moving a `data_struct` to the CPU.                                                                                         | `data_struct`를 CPU로 이동하는 단축 메서드한다. |
| `cuda`                                     | `self`, `ct`                                                                                               | Shortcut for moving a `data_struct` to the GPU(s).                                                                                        | `data_struct`를 GPU로 이동하는 단축 메서드한다. |
| `tensor_device`                            | `self`, `data`                                                                                             | Returns the device type ('cpu' or 'cuda') of the tensor data.                                                                             | 텐서 데이터의 디바이스 유형('cpu' 또는 'cuda')을 반환한다. |
| `device`                                   | `self`, `text`                                                                                             | Returns the device type of a `data_struct` object.                                                                                        | `data_struct` 객체의 디바이스 유형을 반환한다. |
| `tree_lead_text`                           | `self`, `level`, `tabs=2`, `final=False`                                                                   | Generates leading characters for printing a tree-like structure.                                                                          | 트리 형태의 구조를 출력하기 위한 선행 문자를 생성한다. |
| `print_data_shapes`                        | `self`, `data`, `level`                                                                                    | Prints the shapes of tensors within a data structure in a tree format.                                                                    | 데이터 구조 내의 텐서 형태를 트리 형식으로 출력한다. |
| `print_data_structure`                     | `self`, `text`, `level=0`                                                                                  | Prints the structure of a `data_struct` object, including tensor shapes.                                                                  | 텐서 형태를 포함한 `data_struct` 객체의 구조를 출력한다. |
| `auto_generate_filename`                   | `self`, `fmt_str='%Y%m%d%H%M%s%f'`                                                                         | Generates a filename based on the current timestamp.                                                                                      | 현재 타임스탬프를 기반으로 파일 이름을 생성한다. |
| `save`                                     | `self`, `text`, `filename=None`                                                                          | Saves a `data_struct` object to a file using pickle. Moves to CPU if on GPU.                                                                 | `data_struct` 객체를 pickle을 사용하여 파일로 저장합니다. GPU에 있는 경우 CPU로 이동한다. |
| `load`                                     | `self`, `filename`, `move_to_gpu=True`                                                                     | Loads a `data_struct` object from a pickled file. Optionally moves to GPU.                                                                 | pickle된 파일에서 `data_struct` 객체를 로드합니다. 선택적으로 GPU로 이동한다. |
| `negate`                                   | `self`, `ct: data_struct`                                                                                  | Negates a ciphertext `ct`.                                                                                                                 | 암호문 `ct`를 부정한다. |
| `mult_int_scalar`                          | `self`, `ct: data_struct`, `scalar`, `evk=None`, `relin=True`                                             | Multiplies a ciphertext `ct` by an integer scalar.                                                                                        | 암호문 `ct`에 정수 스칼라를 곱한다. |
| `mult_scalar`                              | `self`, `ct`, `scalar`, `evk=None`, `relin=True`                                                         | Multiplies a ciphertext `ct` by a float scalar. Involves rescaling.                                                                       | 암호문 `ct`에 실수 스칼라를 곱한다. 재조정이 포함된다. |
| `add_scalar`                               | `self`, `ct`, `scalar`                                                                                      | Adds a scalar to a ciphertext `ct`.                                                                                                         | 암호문 `ct`에 스칼라를 더한다. |
| `sub_scalar`                               | `self`, `ct`, `scalar`                                                                                      | Subtracts a scalar from a ciphertext `ct`.                                                                                                 | 암호문 `ct`에서 스칼라를 뺀다. |
| `int_scalar_mult`                          | `self`, `scalar`, `ct`, `evk=None`, `relin=True`                                                         | Alias for `mult_int_scalar` with arguments swapped.                                                                                        | 인자 순서를 바꾼 `mult_int_scalar`의 별칭이다. |
| `scalar_mult`                              | `self`, `scalar`, `ct`, `evk=None`, `relin=True`                                                         | Alias for `mult_scalar` with arguments swapped.                                                                                            | 인자 순서를 바꾼 `mult_scalar`의 별칭이다. |
| `scalar_add`                               | `self`, `scalar`, `ct`                                                                                      | Alias for `add_scalar` with arguments swapped.                                                                                                 | 인자 순서를 바꾼 `add_scalar`의 별칭이다. |
| `scalar_sub`                               | `self`, `scalar`, `ct`                                                                                      | Subtracts a ciphertext `ct` from a scalar.                                                                                                 | 스칼라에서 암호문 `ct`를 뺀다. |
| `mc_mult`                                  | `self`, `m`, `ct`, `evk=None`, `relin=True`                                                                 | Multiplies a plaintext message `m` with a ciphertext `ct`.                                                                                 | 평문 메시지 `m`과 암호문 `ct`를 곱한다. |
| `mc_add`                                   | `self`, `m`, `ct`                                                                                             | Adds a plaintext message `m` to a ciphertext `ct`.                                                                                           | 평문 메시지 `m`을 암호문 `ct`에 더한다. |
| `mc_sub`                                   | `self`, `m`, `ct`                                                                                             | Subtracts a ciphertext `ct` from a plaintext message `m`.                                                                                 | 평문 메시지 `m`에서 암호문 `ct`를 뺀다. |
| `cm_mult`                                  | `self`, `ct`, `m`, `evk=None`, `relin=True`                                                                 | Alias for `mc_mult` (ciphertext-message multiplication).                                                                                   | `mc_mult`의 별칭(암호문-메시지 곱셈)이다. |
| `cm_add`                                   | `self`, `ct`, `m`                                                                                             | Alias for `mc_add` (ciphertext-message addition).                                                                                           | `mc_add`의 별칭(암호문-메시지 덧셈)이다. |
| `cm_sub`                                   | `self`, `ct`, `m`                                                                                             | Subtracts a plaintext message `m` from a ciphertext `ct`.                                                                                   | 암호문 `ct`에서 평문 메시지 `m`을 뺀다. |
| `auto_level`                               | `self`, `ct0`, `ct1`                                                                                          | Automatically levels two ciphertexts `ct0` and `ct1` to the same level before an operation.                                                  | 연산 전에 두 암호문 `ct0`와 `ct1`의 레벨을 자동으로 맞춘다. |
| `auto_cc_mult`                             | `self`, `ct0`, `ct1`, `evk`, `relin=True`                                                                     | Performs ciphertext-ciphertext multiplication with automatic leveling.                                                                       | 자동 레벨 조정과 함께 암호문-암호문 곱셈을 수행한다. |
| `auto_cc_add`                              | `self`, `ct0`, `ct1`                                                                                             | Performs ciphertext-ciphertext addition with automatic leveling.                                                                             | 자동 레벨 조정과 함께 암호문-암호문 덧셈을 수행한다. |
| `auto_cc_sub`                              | `self`, `ct0`, `ct1`                                                                                             | Performs ciphertext-ciphertext subtraction with automatic leveling.                                                                           | 자동 레벨 조정과 함께 암호문-암호문 뺄셈을 수행한다. |
| `mult`                                     | `self`, `a`, `b`, `evk=None`, `relin=True`                                                                     | Generic multiplication function that dispatches to the appropriate method based on operand types (ciphertext, scalar, message).                | 피연산자 유형(암호문, 스칼라, 메시지)에 따라 적절한 메서드로 디스패치하는 일반 곱셈 함수이다. |
| `add`                                      | `self`, `a`, `b`                                                                                               | Generic addition function that dispatches to the appropriate method based on operand types.                                                    | 피연산자 유형에 따라 적절한 메서드로 디스패치하는 일반 덧셈 함수이다. |
| `sub`                                      | `self`, `a`, `b`                                                                                               | Generic subtraction function that dispatches to the appropriate method based on operand types.                                                   | 피연산자 유형에 따라 적절한 메서드로 디스패치하는 일반 뺄셈 함수이다. |
| `refresh`                                    | `self`                                                                                                         | Refreshes the internal state of the cryptographically secure pseudo-random number generator (CSPRNG).                                        | 암호학적으로 안전한 의사 난수 생성기(CSPRNG)의 내부 상태를 새로 고친다. |
| `reduce_error`                               | `self`, `ct`                                                                                                   | Reduces accumulated error in a ciphertext `ct` by multiplying by 1.0 (which triggers rescaling).                                               | 1.0을 곱하여(재조정 트리거) 암호문 `ct`의 누적된 오차를 줄인다. |
| `sum`                                        | `self`, `ct`, `gk`                                                                                               | Computes the sum of all elements in a ciphertext `ct` using Galois keys `gk` for rotations.                                                    | 갈루아 키 `gk`를 사용한 회전으로 암호문 `ct`의 모든 원소의 합을 계산한다. |
| `mean`                                       | `self`, `ct`, `gk`, `alpha=1`                                                                                      | Computes the mean of all elements in a ciphertext `ct` using Galois keys `gk`.                                                                 | 갈루아 키 `gk`를 사용하여 암호문 `ct`의 모든 원소의 평균을 계산한다. |
| `cov`                                        | `self`, `ct_a: data_struct`, `ct_b: data_struct`, `evk: data_struct`, `gk: data_struct`                                | Computes the covariance between two ciphertexts `ct_a` and `ct_b` using evaluation key `evk` and Galois keys `gk`.                                | 평가 키 `evk`와 갈루아 키 `gk`를 사용하여 두 암호문 `ct_a`와 `ct_b` 사이의 공분산을 계산한다. |
| `pow`                                        | `self`, `ct: data_struct`, `power: int`, `evk: data_struct`                                                      | Computes `ct` raised to the power of `power` using the evaluation key `evk`.                                                                 | 평가 키 `evk`를 사용하여 암호문 `ct`를 `power` 제곱한다. |
| `square`                                     | `self`, `ct: data_struct`, `evk: data_struct`, `relin=True`                                                      | Computes the square of a ciphertext `ct` using evaluation key `evk`, optionally relinearizing.                                                | 평가 키 `evk`를 사용하여 암호문 `ct`의 제곱을 계산하고, 선택적으로 재선형화한다. |
| `sqrt`                                       | `self`, `ct: data_struct`, `evk: data_struct`, `e=0.0001`, `alpha=0.0001`                                              | Computes the approximate square root of a ciphertext `ct` using Newton's method or similar iterative algorithm.                                | 뉴턴 방법 또는 유사한 반복 알고리즘을 사용하여 암호문 `ct`의 근사 제곱근을 계산한다. |
| `var`                                        | `self`, `ct: data_struct`, `evk: data_struct`, `gk: data_struct`, `relin=False`                                      | Computes the variance of the elements in a ciphertext `ct`.                                                                                | 암호문 `ct`의 원소들의 분산을 계산한다. |
| `std`                                        | `self`, `ct: data_struct`, `evk: data_struct`, `gk: data_struct`, `relin=False`                                      | Computes the standard deviation of the elements in a ciphertext `ct`.                                                                      | 암호문 `ct`의 원소들의 표준편차를 계산한다. |

---

## dmesg Analysis for main.py Execution Errors

Analysis of the `dmesg_output.txt` reveals critical errors related to the NVIDIA GPU and its driver, which are likely causes for `main.py` execution failures, especially since `main.py` utilizes CUDA.

Key errors observed:

1.  **NVIDIA Xid Errors (MMU Faults):**
    *   Log entries like `NVRM: Xid (PCI:0000:01:00): 31, pid='<unknown>', name=<unknown>, Ch 00000008, intr 00000000. MMU Fault: ENGINE GRAPHICS GPCCLIENT_T1_0 faulted @ 0x0_40674000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ` appear multiple times.
    *   Xid 31 errors generally indicate serious problems with the GPU, often related to the Memory Management Unit (MMU). This can be due to:
        *   Hardware issues with the GPU.
        *   NVIDIA driver problems (corruption, incompatibility).
        *   Overheating of the GPU.
        *   An unstable GPU state.
    *   These MMU faults mean the GPU cannot correctly manage or access its memory, which is fundamental for CUDA operations.

2.  **DRM (Direct Rendering Manager) Errors:**
    *   Log entries such as `[drm:nv_drm_master_set [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to grab modeset ownership` are also present.
    *   These errors suggest that the NVIDIA display driver is struggling to take or maintain control over the GPU's display modes and general operation.

**Conclusion:**

The combination of NVRM Xid 31 MMU faults and DRM errors points to a severe instability or malfunction at the GPU hardware or driver level. Consequently, the CUDA runtime environment required by `main.py` would likely fail to initialize correctly, or any CUDA operations would fail, leading to the execution errors observed in `main.py`.

Troubleshooting steps should focus on:
*   Ensuring the NVIDIA drivers are correctly installed and are compatible with the kernel version and GPU.
*   Checking the GPU for overheating.
*   Testing the GPU hardware for potential faults.

## main.py 실행 오류에 대한 dmesg 분석

`dmesg_output.txt` 분석 결과, NVIDIA GPU 및 드라이버와 관련된 심각한 오류가 발견되었습니다. 이는 CUDA를 사용하는 `main.py` 실행 실패의 가능성이 높은 원인입니다.

### 발견된 주요 오류:

**1. NVIDIA Xid 오류(MMU 폴트):**
* `NVRM: Xid (PCI:0000:01:00): 31, pid='<unknown>', name=<unknown>, Ch 00000008, intr 00000000. MMU Fault: ENGINE GRAPHICS GPCCLIENT_T1_0 faulted @ 0x0_40674000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ`와 같은 로그가 여러 번 발생했습니다.
* Xid 31 errors generally indicate serious problems with the GPU, often related to the Memory Management Unit (MMU). 다음과 같은 원인이 있을 수 있습니다:
  * GPU 하드웨어 문제
  * NVIDIA 드라이버 문제(손상, 비호환성)
  * GPU 과열
  * 불안정한 GPU 상태
* 이러한 MMU 폴트는 GPU가 메모리를 올바르게 관리하거나 접근할 수 없음을 의미하며, 이는 CUDA 작업에 필수적입니다.

**2. DRM(Direct Rendering Manager) 오류:**
* `[drm:nv_drm_master_set [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00000100] Failed to grab modeset ownership`와 같은 로그도 존재합니다.
* 이러한 오류는 NVIDIA 디스플레이 드라이버가 GPU의 디스플레이 모드와 일반 작동에 대한 제어권을 확보하거나 유지하는 데 어려움을 겪고 있음을 시사합니다.

### 결론:

NVRM Xid 31 MMU 폴트와 DRM 오류의 조합은 GPU 하드웨어 또는 드라이버 수준에서 심각한 불안정성이나 오작동이 있음을 나타냅니다. 결과적으로 `main.py`에 필요한 CUDA 런타임 환경이 제대로 초기화되지 않거나 CUDA 작업이 실패하여 `main.py` 실행 오류가 발생했을 가능성이 높습니다.

### 문제 해결 단계:

* NVIDIA 드라이버가 올바르게 설치되었고 커널 버전 및 GPU와 호환되는지 확인
* GPU 과열 여부 확인
* GPU 하드웨어의 잠재적 결함 테스트

---

## PowerformerEncoder 구현 상세

### 헬퍼 함수

#### `_encode_col_blocks(mat_head1, mat_head2, engine, level=0)`
- **입력**:
  - `mat_head1`: 첫 번째 헤드 행렬 (numpy 배열)
  - `mat_head2`: 두 번째 헤드 행렬 (numpy 배열)
  - `engine`: CKKS 엔진 인스턴스
  - `level`: 인코딩 레벨 (기본값=0)
- **출력**: 인코딩된 평문 블록 리스트
- **설명**: 두 헤드 행렬을 효율적인 병렬 처리를 위해 인터리브된 평문 블록으로 인코딩

#### `load_layer_weights(layer_dir: str, engine) -> dict`
- **입력**: 
  - `layer_dir`: 가중치 파일이 있는 디렉토리 경로
  - `engine`: CKKS 엔진 인스턴스
- **출력**: 인코딩된 가중치와 매개변수를 포함하는 딕셔너리
- **설명**: .npy 파일에서 레이어 가중치를 로드하고 FHE 연산을 위한 평문 형식으로 인코딩

### PowerformerEncoder 클래스

#### 초기화
```python
def __init__(self, engine, evk, params, w)
```
- **매개변수**:
  - `engine`: CKKS 엔진 for homomorphic operations
  - `evk`: 평가 키 번들 (회전 및 재선형화 키 포함)
  - `params`: Dictionary containing BRPmax/PolyReLU/LayerNorm constants
  - `w`: Dictionary containing encoded weight matrices
- **사전 인코딩된 상수**: 
  - BRPmax 매개변수
  - ReLU 다항식 계수
  - LayerNorm 상수

#### 핵심 메서드

##### `__call__(self, ctx_in)`
- **입력**: Ciphertext with (X1/X2) packed 128×128 format
- **출력**: Processed ciphertext for next block
- **처리 흐름**:
  1. 멀티헤드 어텐션
  2. Feed-Forward with PolyReLU
  3. Optional bootstrapping
  
##### 행렬 곱셈 메서드

###### `_constrec_mm(self, ct_in, pt_blocks)`
- **목적**: 상수-재귀 행렬 곱셈
- **입력**:
  - `ct_in`: Ciphertext (X1/X2) packed 128×128
  - `pt_blocks`: 평문 블록 리스트 (각각 2×128)
- **출력**: (Q1/Q2) packed 128×128
- **알고리즘**: 효율적인 행렬 곱셈을 위한 베이비스텝-자이언트스텝 최적화 사용

###### `_const_mm(self, ct_in, pt_W)`
- **목적**: 상수 행렬 곱셈
- **입력**:
  - `ct_in`: Ciphertext (X1/X2) row-major interleaved
  - `pt_W`: 평문 (W1/W2 interleaved)
- **출력**: (Y1/Y2) maintaining even/odd structure
- **최적화**: 반복 연산을 위한 마스크 캐싱 사용

###### `_block_mm_core(self, ct_A, ct_B_T)`
- **목적**: 코어 블록 행렬 곱셈
- **입력**:
  - `ct_A`: 행렬 A 암호문 (d x 2k)
  - `ct_B_T`: 전치된 행렬 B 암호문 (2k x d')
- **출력**: A * B_T 곱
- **특징**: 헤드 인터리브 형식 및 분할 계산 처리

##### 변환 메서드

###### `_transpose(self, ct_in)`
- **목적**: 암호문 공간에서의 행렬 전치
- **입력**: 행 우선 (X/X) ciphertext
- **출력**: 전치된 (X/X) ciphertext
- **최적화**: 캐시된 대각선 마스크 사용

##### 활성화 및 정규화

###### `_brpmax_poly(self, ct)`
- **목적**: 유계 ReLU 다항식 최대값
- **알고리즘**: ReLU(x+c)^p / R_d
- **지원 거듭제곱**: p=2 또는 p=3

###### `_poly_relu(self, ct)`
- **목적**: ReLU의 다항식 근사
- **공식**: a1·x + a3·x³
- **매개변수**: 사전 인코딩된 계수 사용

###### `_layernorm_poly(self, ct, ln_pt)`
- **목적**: 레이어 정규화
- **알고리즘**: γ · (x−μ) · inv_sqrt(var+ε) + β
- **특징**:
  - 행 단위 평균 및 분산
  - 역제곱근을 위한 뉴턴 방법
  - 헤드 인터리브 형식 유지

##### 유틸리티 메서드

###### `_bootstrap_if_needed(self, ct)`
- **목적**: 딥 네트워크를 위한 레벨 관리
- **동작**: 조건부 부트스트래핑 또는 레벨 조정 수행

### 주요 특징 및 최적화

1. **효율적인 데이터 패킹**
   - 병렬 처리를 위한 인터리브된 헤드 형식
   - 전략적 패킹을 통한 최적화된 메모리 사용

2. **성능 최적화**
   - 반복 연산을 위한 캐시된 마스크
   - 행렬 곱셈을 위한 베이비스텝-자이언트스텝 알고리즘
   - 비선형 함수를 위한 효율적인 다항식 근사

3. **FHE 친화적 설계**
   - 비선형 함수를 위한 다항식 근사
   - 준동형 환경에 최적화된 행렬 연산
   - 부트스트래핑 지원이 있는 레벨 인식 연산

4. **메모리 관리**
   - 빈번한 연산을 위한 사전 인코딩된 상수
   - 계산 마스크의 캐싱
   - 블로킹을 통한 대형 행렬의 효율적인 처리

### 구현 참고사항

1. **행렬 차원**
   - 표준 입력 크기: 128×128 행렬
   - 헤드 분할: 2개의 어텐션 헤드
   - 블록 크기: √d 연산에 최적화

2. **수치 안정성**
   - LayerNorm은 2단계 뉴턴 반복 사용
   - 신중하게 선택된 다항식 근사
   - 재조정 연산을 통한 스케일 관리

3. **오류 처리**
   - 다항식 차수 검증
   - 딥 네트워크를 위한 레벨 모니터링
   - 행렬 연산을 위한 적절한 형상 검사

---

## Client.py 구현 분석

### 개요
`client.py`는 추론할 문장을 토큰화하고 임베딩하여 Powerformer 서버로 전송하는 역할을 한다. 클라이언트는 Liberate 엔진을 사용해 암호화 키(sk, pk, evk)를 생성하고, 추론할 문장을 토큰화 및 임베딩한 후 이를 pk, evk와 함께 서버로 전송한다.

### 주요 구성 요소

#### 1. 라이브러리 의존성
```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import liberate.fhe as fhe
```

#### 2. 유틸리티 함수

##### `pack_parallel(matA: np.ndarray, matB: np.ndarray | None = None) -> np.ndarray`
- **목적**: 두 행렬을 짝-홀 슬롯에 교차 삽입하여 병렬 처리를 위한 벡터 생성
- **동작 방식**:
  - 입력 행렬을 row-major 순서로 평탄화
  - 짝수 슬롯에는 matA의 값을 배치
  - 홀수 슬롯에는 matB의 값을 배치 (matB가 None인 경우 matA를 재사용)
  - 허수부는 0으로 설정
- **반환**: 복소수 타입(np.complex128)의 병렬 패킹된 벡터

#### 3. 핵심 함수

##### `prepare_client_data()`
- **목적**: 클라이언트 데이터 준비 및 암호화
- **주요 단계**:
  1. FHE 엔진 초기화
     ```python
     params = fhe.presets.params["gold"].copy()
     params["devices"] = ['cpu']
     engine = fhe.ckks_engine(**params, verbose=True)
     ```
  
  2. 암호화 키 생성
     ```python
     sk = engine.create_secret_key()
     pk = engine.create_public_key(sk)
     evk = engine.create_evk(sk)  # 회전·relin·bootstrapping 키 묶음
     ```
  
  3. BERT 모델 및 토크나이저 초기화
     ```python
     tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
     bert_tiny = AutoModel.from_pretrained("prajjwal1/bert-tiny", add_pooling_layer=False)
     ```
  
  4. 데이터 준비 및 처리
     - RTE 데이터셋에서 검증 데이터 샘플 로드
     - 문장 쌍 토큰화 (max_length=128)
     - BERT 임베딩 생성
  
  5. 데이터 암호화
     - 임베딩을 병렬 패킹
     - 패킹된 데이터 인코딩
     - 인코딩된 데이터 암호화

- **반환값**: (engine, ctx_in, pk, evk)
  - engine: FHE 엔진 인스턴스
  - ctx_in: 암호화된 입력 데이터
  - pk: 공개키
  - evk: 평가키

### 데이터 흐름
1. 문장 입력 → 토큰화 → BERT 임베딩
2. 임베딩 → 병렬 패킹 → FHE 인코딩
3. 인코딩된 데이터 → FHE 암호화
4. 암호화된 데이터 + 키 → 서버 전송

### 보안 특징
- 비밀키(sk)는 클라이언트에만 존재
- 공개키(pk)와 평가키(evk)만 서버로 전송
- 데이터는 항상 암호화된 상태로 전송 및 처리

### 최적화 포인트
- CPU 기반 처리 (`params["devices"] = ['cpu']`)
- 효율적인 병렬 패킹을 통한 데이터 처리
- BERT-tiny 모델 사용으로 경량화
- 128 토큰으로 제한된 입력 길이

---

## Server.py 구현 분석

### 개요
`server.py`는 클라이언트로부터 암호화된 입력 데이터와 필요한 키들을 받아 FHE 기반의 Powerformer 추론을 수행하는 서버 측 구현이다. 이 모듈은 모델 구성을 로드하고, 레이어별로 암호화된 연산을 수행하여 최종 결과를 반환한다.

### 주요 구성 요소

#### 1. 전역 설정
```python
MODEL_DIR = "./student_powerformer_rte"
CONFIG_FILE = "powerformer_encoder_config.json"
```
- `MODEL_DIR`: Powerformer 모델 가중치와 설정이 저장된 디렉토리 경로
- `CONFIG_FILE`: 모델 구성 정보가 담긴 JSON 파일명

#### 2. 핵심 함수

##### `handle_client_data(engine_instance, encrypted_input, public_key, eval_key)`
- **목적**: 클라이언트로부터 받은 암호화된 데이터에 대해 FHE 기반 추론 수행
- **매개변수**:
  - `engine_instance`: CKKS 엔진 인스턴스
  - `encrypted_input`: 암호화된 입력 데이터
  - `public_key`: 공개키
  - `eval_key`: 평가키 (회전, 재선형화, 부트스트래핑 키 포함)
- **주요 처리 단계**:
  1. 모델 설정 로드
     - config.json 파일에서 모델 구성 읽기
     - 레이어 수 및 활성화 함수 매개변수 확인
  
  2. 인코더 레이어 초기화
     - 각 레이어별 가중치 로드
     - PowerformerEncoder 인스턴스 생성
     - 레이어 체인 구성
  
  3. 순차적 추론 수행
     - 각 레이어를 통과하며 암호화된 연산 수행
     - 중간 결과를 다음 레이어로 전달
  
- **반환값**: 
  - 성공 시: 최종 암호화된 출력
  - 실패 시: None

### 오류 처리
1. 설정 파일 검증
   ```python
   if not os.path.exists(config_path):
       print(f"ERROR: Powerformer config file not found at {config_path}")
       return None
   ```

2. 필수 설정 매개변수 확인
   ```python
   if num_hidden_layers is None:
       print(f"ERROR: 'num_hidden_layers' not found in powerformer_encoder_config.json")
       return None
   ```

3. 레이어 디렉토리 존재 확인
   ```python
   if not os.path.isdir(layer_dir):
       print(f"ERROR: Layer weights directory not found: {layer_dir}")
       return None
   ```

### 데이터 흐름
1. 클라이언트 → 서버: 암호화된 입력 + 키
2. 서버: 모델 설정 및 가중치 로드
3. 서버: 레이어별 순차 처리
4. 서버 → 클라이언트: 암호화된 출력

### 보안 특징
- 모든 연산이 암호화된 상태에서 수행
- 클라이언트의 비밀키 없이 연산 가능
- 중간 결과도 항상 암호화 상태 유지

### 개발자 참고사항
- 직접 실행 불가능 (클라이언트 제공 FHE 객체 필요)
- 테스트를 위한 모의 객체 예제 코드 포함
- 모델 디렉토리 구조 준수 필요
  ```
  student_powerformer_rte/
  ├── powerformer_encoder_config.json
  ├── layer00/
  │   ├── WQ.npy
  │   └── ...
  ├── layer01/
  │   └── ...
  └── ...
  ```


# powerformer-liberate
