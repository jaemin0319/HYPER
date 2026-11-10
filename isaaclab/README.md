# Isaac Lab 학습 환경 (HYPER)

Isaac Lab 2.3의 Unitree Go2 속도 추종 태스크(`Isaac-Velocity-Flat/Rough-Unitree-Go2-v0`)를 HYPER에 맞게 가져왔습니다. 설정 클래스는 Isaac Lab의 Go2 클래스를 **상속**하므로, 아래 표에 적은 것 말고는 지형, 명령, 액션, 시뮬레이션 주기, 이벤트(도메인 랜덤화), 보상, 종료, 커리큘럼, PPO 설정이 모두 Go2와 같습니다.

| 태스크 | 내용 |
| --- | --- |
| `Isaac-Velocity-Flat-HYPER-v0` / `-Play-v0` | 평지. 학습 완료 정책: `deploy/policies/flat/` |
| `Isaac-Velocity-Rough-HYPER-v0` / `-Play-v0` | Go2 rough 지형(작은 로봇용으로 축소). 환경 점검만 했고 학습은 하지 않음 |

## Go2와 다른 점 (모두 하드웨어 때문)

| 항목 | Go2 (Isaac Lab) | HYPER | 이유 |
| --- | --- | --- | --- |
| 로봇 | `go2.usd` | `usd/hyper.usd` | |
| 모터 | `DCMotorCfg` 23.5 N·m, 30 rad/s | `DCMotorCfg` 23.7 N·m, 30 rad/s (kp 25, kd 0.5 동일) | GO-M8010-6, calf는 thigh에 달린 모터가 벨트 1:1 |
| 정책 관측 | 몸체 선속도, 각속도, 중력 방향, 명령, q, dq, 직전 액션 (+ rough: height scan) | 명령(3), q − q_default(12), dq(12), 직전 액션(12) = 39 | IMU·상태추정기·라이다·발 센서가 없음. 로봇에서 재는 건 모터 엔코더뿐 |
| critic 관측 | 정책 관측과 같음 | Go2 정책 관측 전체(48, rough는 +187 height scan) | 시뮬레이션에서만 쓰는 비대칭 actor-critic(rsl_rl `obs_groups`). 로봇에는 올라가지 않음 |
| 자기충돌 | 끔 | **켬** | 관절 범위 안에서 허벅지가 몸체에 닿음(CAD 간섭: 앞 허벅지 약 1.40 rad, hip 외전 약 0.85 rad) |
| `undesired_contacts` | 끔 | 허벅지, 가중치 −1, 1 N (Isaac Lab 기본 locomotion 설정의 항목) | 허벅지를 몸체·바닥에 기대 걷지 않도록 |
| `base_contact` 종료 | 몸체 접촉력 합 > 1 N | 몸체-**지면** 접촉력 > 1 N (지면으로 필터한 접촉 센서) | 자기충돌을 켜면 허벅지-몸체 접촉도 합에 들어가 '넘어짐'으로 잘못 판정됨 |

관측 노이즈(q ±0.01, dq ±1.5), 액션 스케일 0.25, 정책 50 Hz(dt 0.005 × decimation 4), 명령 범위(±1 m/s, ±1 rad/s)는 Go2와 같습니다.

### 위 세 줄(자기충돌·허벅지 페널티·종료)을 바꾼 과정

1. Go2 설정을 그대로 쓴 첫 학습은 Isaac에서는 잘 걸었습니다(명령 10종에서 넘어짐 0회). 하지만 MuJoCo sim2sim에서는 1 m/s 명령에 0.56 m/s만 나오고, 몸이 31°까지 기울고, 회전 명령에는 돌지 않았습니다. 원인은 정책이 웅크린 자세(앞 허벅지 1.14~1.98 rad)를 배운 것이었습니다. 이 자세에서는 앞 허벅지가 몸체를 통과하고, MuJoCo에서는 그 접촉 때문에 막혔습니다. 실기에서도 같은 일이 생깁니다.
2. 자기충돌만 켜면 Go2의 `base_contact`가 허벅지-몸체 접촉에도 반응해서, 정책이 3스텝 만에 에피소드를 끝내는 쪽으로 무너졌습니다.
3. 종료 조건을 몸체-지면 접촉만 보도록 바꾸고 허벅지 접촉은 페널티로 처리해서 다시 학습했습니다(아래 결과).

## 설치·실행

Isaac Sim 5.1 + Isaac Lab 2.3이 이미 있는 Python 위에 `.venv`를 얹습니다. 원래 환경은 바꾸지 않습니다.

```bash
ISAAC_PYTHON=/path/to/isaac/env/bin/python ISAACLAB_PATH=/path/to/IsaacLab bash isaaclab/setup_venv.sh
source isaaclab/activate.sh            # HYPER_GPU=<index|uuid> 로 GPU 선택

python isaaclab/scripts/check_env.py --task Isaac-Velocity-Flat-HYPER-v0          # 관측·액추에이터·MDP 출력, 0 액션으로 서기
python isaaclab/scripts/train.py --task Isaac-Velocity-Flat-HYPER-v0 --headless   # Isaac Lab rsl_rl train.py 그대로
python isaaclab/scripts/play.py  --task Isaac-Velocity-Flat-HYPER-Play-v0 --num_envs 16
python isaaclab/scripts/export_deploy.py --task Isaac-Velocity-Flat-HYPER-Play-v0 \
    --checkpoint logs/rsl_rl/hyper_flat/<run>/model_299.pt --out deploy/policies/flat
```

`train.py`·`play.py`는 Isaac Lab의 `scripts/reinforcement_learning/rsl_rl/` 스크립트를 HYPER 태스크를 등록한 채로 그대로 실행합니다. `export_deploy.py`는 Isaac Lab exporter로 `policy.pt`·`policy.onnx`를 만들고, 배포용 `policy.npz`·`deploy.json`을 쓴 뒤 배포 코드와 Isaac Lab이 같은지, 그리고 명령 추종이 어떤지 검증합니다.

## 결과 (평지 정책 `deploy/policies/flat/`)

`Isaac-Velocity-Flat-HYPER-v0`, env 4096개, 300 iteration(Go2 flat 기본값 그대로). 체크포인트 `model_299.pt`를 export했습니다.

**배포 코드 = Isaac Lab** (`export_deploy.py`, 같은 rollout에서 비교, `isaac_eval.json`): 관측 오차 0, 액션 2.1e-6, 관절 목표 1.2e-7, DC 모터 토크 1.6e-6.

**명령 추종** (평균 속도, 몸체 좌표). Isaac은 명령마다 env 5개 × 10초(처음 3초 제외)로 쟀습니다. MuJoCo는 `deploy/scripts/run_sim2sim.py`로 배포 코드를 그대로 돌려 쟀고, 명령 10개를 이어서 약 62초 동안 실행했습니다.

| 명령 (vx, vy, wz) | Isaac Lab | MuJoCo 마찰 1.0 | MuJoCo 마찰 0.6 |
| --- | --- | --- | --- |
| 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| vx 0.5 | 0.51 | 0.46 | 0.52 |
| vx 1.0 | 0.96 | 0.91 | 0.98 |
| vx −0.5 | −0.51 | −0.49 | −0.52 |
| vy 0.5 | 0.53 | 0.50 | 0.55 |
| vy −0.5 | −0.52 | −0.47 | −0.53 |
| wz 1.0 | 0.65 | 0.63 | 0.65 |
| wz −1.0 | −0.87 | **0 (돌기 시작하지 못함)** | −0.93 |
| vx 0.5 + wz 0.5 | 0.54, 0.47 | 0.47, 0.41 | 0.55, 0.48 |
| vx 0.7 + vy 0.4 | 0.72, 0.44 | 0.64, 0.39 | 0.72, 0.45 |

어느 조건에서도 넘어지지 않았습니다. 최대 기울기는 Isaac 7.5°, MuJoCo 7.1°이며, 둘 다 wz 1.0에서 나왔습니다.

### 알려진 약점 (재학습 전 상태로 공개)

- **제자리 회전이 약함**: 1 rad/s 명령에 0.63~0.93 rad/s만 돕니다. MuJoCo 마찰 1.0에서는 서 있다가 −1.0 rad/s 명령을 받으면 돌기 시작하지 못하고, −0.5 rad/s도 마찰 1.0·0.8에서 같습니다. 바닥 마찰에 따라 결과가 달라집니다.
- **wz +1.0에서 오른쪽 앞 허벅지가 몸체에 닿음**: MuJoCo 구간 시간의 12~27 %.
- **발을 거의 들지 않음**: MuJoCo에서 잰 발 높이는 1~17 mm로, 끌듯이 걷습니다. 문턱이나 카펫 같은 작은 턱에 걸릴 수 있습니다. 1 m/s 직진에서는 발이 아닌 부분이 바닥에 닿는 시간이 8~9 %입니다.
- 추정 원인은 세 가지입니다.
  1. Go2의 `feet_air_time` 보상은 xy 명령이 0.1을 넘을 때만 주어집니다. 그래서 제자리 회전에서는 발을 들 이유가 없습니다.
  2. 마찰이 고정값입니다(Go2 설정 그대로).
  3. 300 iteration으로 짧게 학습했습니다.
- 개선 방향: `feet_air_time`에 회전 명령 포함, 발 높이 보상 추가, 마찰 랜덤화, 1500 iteration 재학습.
- Rough 태스크는 학습하지 않았고, 정책은 실기에서 아직 검증하지 않았습니다.
