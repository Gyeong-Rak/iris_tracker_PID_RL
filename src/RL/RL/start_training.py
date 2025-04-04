#!/usr/bin/env python3
import subprocess
import time
import os
import signal

def run_command(command, cwd=None, wait=False, verbose=False):
    print(f"Running: {command}")

    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        preexec_fn=os.setsid,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL
    )
    if wait:
        process.wait()
    return process

def main():
    Done = False
    episode = 0
    max_episode = 50
    ws_dir = os.path.join(os.getcwd(), "iris_tracker_PID_RL")
    processes = []

    if episode > max_episode:
        print("Model training complete")
        for proc in processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                print(f"종료 중 오류: {e}")
        print("모든 프로세스를 종료했습니다.")

    try:
        while not Done:
            episode += 1
            print(f"\n--- Training Episode {episode} 시작 ---\n")
            
            # 1. 시뮬레이션 실행 (GUI 없이)
            sim_cmd = "./PX4-Autopilot_v1.15.4/Tools/simulation/gazebo-classic/sitl_multiple_run2.sh -s \"iris_camera:1,iris:1\" -w test_world1"
            sim_process = run_command(sim_cmd)
            processes.append(sim_process)
            time.sleep(10)  # 시뮬레이터 초기화 대기
            
            # 2. MicroXRCE-DDS 실행
            microxrce_cmd = "MicroXRCEAgent udp4 -p 8888"
            microxrce_process = run_command(microxrce_cmd)
            processes.append(microxrce_process)
            
            # 3. yolo_detector node 실행
            yolo_cmd = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run utils yolo_detector'"
            yolo_process = run_command(yolo_cmd, cwd=ws_dir)
            processes.append(yolo_process)

            # 4. fastsam_detector node 실행
            fastsam_cmd = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run utils fastSAM_detector'"
            fastsam_process = run_command(fastsam_cmd, cwd=ws_dir)
            processes.append(fastsam_process)
            time.sleep(10)  # wait for fastsam

            # 5. iris_controller node 실행
            controller_cmd = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run controller iris_controller'"
            iris_controller_process = run_command(controller_cmd, cwd=ws_dir)
            processes.append(iris_controller_process)

            # 6. iris_camera_controller_PID node 실행
            controller_cmd = f"bash -c 'source ~/.bashrc && source ./install/local_setup.bash && export PYTHONPATH=$PYTHONPATH:{ws_dir}/src/RL/RL && ros2 run controller iris_camera_controller_PID --ros-args -p mode:=pixel'"
            iris_camera_controller_process = run_command(controller_cmd, cwd=ws_dir, verbose=True)
            processes.append(iris_camera_controller_process)

            # # 6. iris_camera_controller_RL node 실행
            # controller_cmd = f"bash -c 'source ~/.bashrc && source ./install/local_setup.bash && export PYTHONPATH=$PYTHONPATH:{ws_dir}/src/RL/RL && ros2 run controller iris_camera_controller_RL --ros-args -p mode:=pixel'"
            # iris_camera_controller_process = run_command(controller_cmd, cwd=ws_dir, verbose=True)
            # processes.append(iris_camera_controller_process)
            
            # # 6. 모델 학습
            # training_cmd = f"bash -c 'source ~/.bashrc && source ./install/local_setup.bash && export PYTHONPATH=$PYTHONPATH:{ws_dir}/src/RL/RL && ros2 run RL RL_online_train --ros-args -p max_episodes:={max_episode} -p episode:={episode} -p mode:=pixel'"
            # train_process = run_command(training_cmd, cwd=ws_dir, verbose=True)
            # processes.append(train_process)

            while True:
                try:
                    if os.path.exists("/tmp/rl_episode_done.flag"):
                        print("종료 플래그 감지됨. 학습 종료.")
                        os.remove("/tmp/rl_episode_done.flag")
                        break
                    time.sleep(1)

                except KeyboardInterrupt:
                    print("사용자에 의해 학습 중단됨")
                    Done = True
                    break
            
            for proc in processes:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception as e:
                    print(f"프로세스 종료 중 오류: {e}")
            processes.clear()
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.")
        for proc in processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                print(f"종료 중 오류: {e}")
        print("모든 프로세스를 종료했습니다.")

if __name__ == "__main__":
    main()
