# HYPER 배포 제어 코드

Isaac Lab에서 학습한 속도 추종 정책을 실제 HYPER(GO-M8010-6 × 12)와 MuJoCo(sim2sim)에서 같은 코드로 돌립니다. HYPER에는 IMU·발 센서·라이다가 없으므로 **입력은 모터 엔코더(q, dq)와 조이스틱 명령뿐**입니다. numpy만 있으면 돌아가고, MuJoCo(sim2sim)와 `unitree_actuator_sdk`(실기)는 쓸 때만 필요합니다.

| 파일 | 내용 |
| --- | --- |
| `hyper_deploy/policy_io.py` | 관측 구성, 액션→관절 목표, DC 모터 PD·토크 제한, numpy MLP 정책. Isaac Lab과 같은 식 |
| `hyper_deploy/controller.py` | 상태 기계(passive → stand → walk, 고장 시 damp)와 50 Hz 제어 루프 |
| `hyper_deploy/backends.py` | `MujocoBackend`(sim2sim), `GoM8010Backend`(실기, RS-485 포트마다 IO 스레드) |
| `hyper_deploy/commands.py` | 리눅스 조이스틱(`/dev/input/js*`), MuJoCo 뷰어 키보드, 고정 스크립트 명령 |
| `hyper_deploy/fake_unitree_sdk.py` | 하드웨어 없이 시험하는 가짜 SDK(공식 Python 바인딩과 같은 이름) |
| `policies/flat/` | 학습된 정책: `policy.npz`(배포용), `policy.pt`·`policy.onnx`(Isaac Lab export), `deploy.json`, `isaac_eval.json`, `sim2sim*.json` |
| `config/hardware.example.json` | 모터 배선·방향·영점, 조이스틱 매핑, 안전 설정 템플릿 |
| `scripts/run_sim2sim.py`, `run_real.py`, `calibrate.py` | 실행 스크립트 |
| `tests/test_deploy.py` | 하드웨어 없이 도는 단위 테스트 |

## 제어 루프 (Isaac Lab과 같음)

`deploy.json`은 `isaaclab/scripts/export_deploy.py`가 실제 Isaac Lab 환경에서 값을 읽어 씁니다. 숫자를 손으로 옮기지 않습니다.

매 0.02 s(Isaac Lab 정책 주기, physics 0.005 s × decimation 4)마다:

1. `q, dq` ← 모터 엔코더 (관절 순서: `FL/FR/RL/RR_hip`, `..._thigh`, `..._calf`, Isaac Lab 순서)
2. `obs = [vx, vy, wz, q − q_default, dq, last_action]` (39차원, 스케일·클립 없음)
3. `a = policy(obs)`
4. `q* = q_default + 0.25·a` (Isaac Lab `JointPositionAction`, 실기에서는 관절 한계로 자름)
5. 모터 PD: `τ = 25(q* − q) − 0.5·dq`, GO-M8010-6 토크-속도 제한(정지 23.7 N·m, 무부하 30 rad/s; Isaac Lab `DCMotor`)

조이스틱 명령 범위는 학습과 같습니다: vx, vy ±1.0 m/s, wz ±1.0 rad/s.

## 관절 ↔ 모터

```
q_joint = direction · q_motor_output + offset        (calf: thigh에 달린 모터가 벨트 1:1로 무릎 구동, 같은 식)
q_motor_output = q_rotor / 6.33
```

SDK 명령은 로터 기준입니다([unitree_actuator_sdk README](https://github.com/unitreerobotics/unitree_actuator_sdk)): `q·6.33`, `kp/6.33²`, `kd/6.33²`, `τ/6.33`. PD는 모터 안에서 돌고(FOC), 이 코드는 목표각과 게인만 보냅니다. 테스트가 이 변환에서 관절 기준 강성이 정확히 25 N·m/rad가 되는지 확인합니다.

## 실행

```bash
# 단위 테스트 (하드웨어·Isaac 불필요)
python deploy/tests/test_deploy.py

# sim2sim: MuJoCo 위에서 실기와 같은 컨트롤러, 고정 명령으로 추종·넘어짐 측정
python deploy/scripts/run_sim2sim.py --policy deploy/policies/flat
python deploy/scripts/run_sim2sim.py --policy deploy/policies/flat --viewer --keyboard   # W/S A/D Q/E, ENTER 걷기

# 실기 코드 경로를 가짜 모터로 끝까지 실행
python deploy/scripts/run_real.py --dry-run
```

## 실기 (하드웨어에서는 아직 검증하지 않음)

1. [unitree_actuator_sdk](https://github.com/unitreerobotics/unitree_actuator_sdk)를 받아 `lib/`의 `unitree_actuator_sdk*.so`가 로봇 PC의 Python 버전과 맞는지 확인합니다(필요하면 다시 빌드).
2. `config/hardware.example.json`을 `config/hardware.json`으로 복사하고 포트·모터 ID를 배선에 맞춥니다.
3. 방향 확인: `sudo -E python deploy/scripts/calibrate.py --sdk <lib> --watch`로 관절을 손으로 움직여 URDF 방향(hip +x, thigh·calf +y)과 부호가 맞는지 보고, 틀리면 `direction`을 −1로 바꿉니다.
4. 영점: `calibration_pose`(기본은 CAD 조립 자세)를 손이나 지그로 맞춘 뒤 `calibrate.py --save`. 모터 위치가 전원을 켤 때마다 0부터 시작하는 방식이라면 매번 다시 합니다.
5. 로봇을 매달거나 받침에 올린 상태에서 `sudo -E python deploy/scripts/run_real.py --sdk <lib>`를 실행합니다. START로 2초 동안 기본 자세까지 일어서고, A를 누르면 걷기, B를 누르면 감쇠 정지(복귀 불가)입니다.

안전장치: 모터 오류(과열·과전류·과전압·엔코더), 온도 80 °C, 0.1 s 동안 응답 없음, 관절 한계 0.1 rad 초과 중 하나라도 생기면 damp(kp 0, kd 1)로 넘어갑니다. Ctrl+C를 누르면 1초 동안 감쇠한 뒤 토크를 0으로 만듭니다.
