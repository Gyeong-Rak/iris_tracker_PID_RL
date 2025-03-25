#!/usr/bin/env python3
import subprocess
import time
import os
import signal

def run_command(command, cwd=None, wait=False):
    """
    주어진 커맨드를 실행합니다.
    preexec_fn=os.setsid를 사용해 자식 프로세스 그룹을 생성하므로,
    이후 전체 그룹을 종료할 수 있습니다.
    """
    print(f"Running: {command}")  # 실행 로그 출력
    process = subprocess.Popen(
        command, 
        shell=True, 
        cwd=cwd, 
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,  # 실행 로그 숨김
        stderr=subprocess.DEVNULL   # 오류 로그 숨김
    )
    if wait:
        process.wait()
    return process

def main():
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n--- Training Iteration {iteration} 시작 ---\n")
            
            # 1. 시뮬레이션 실행 (GUI 없이)
            sim_command = "./PX4-Autopilot_v1.15.4/Tools/simulation/gazebo-classic/sitl_multiple_run2.sh -s \"iris_camera:1,iris:1\" -w test_world1"
            sim_process = run_command(sim_command)
            time.sleep(10)  # 시뮬레이터 초기화 대기
            
            # 2. MicroXRCE-DDS 실행
            microxrce_command = "MicroXRCEAgent udp4 -p 8888"
            microxrce_process = run_command(microxrce_command)
            
            # 3. yolo_detector node 실행
            tracker_dir = os.path.join(os.getcwd(), "iris_tracker_PID_RL")
            yolo_command = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run utils yolo_detector'"
            yolo_process = run_command(yolo_command, cwd=tracker_dir)
            
            # 4. iris_controller node 실행
            controller_command = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run controller iris_controller'"
            iris_controller_process = run_command(controller_command, cwd=tracker_dir)
            time.sleep(2)  # 노드 초기화 대기

            # 5. iris_camera_controller node 실행
            controller_command = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run controller iris_camera_controller_PID_setpoint'"
            iris_camera_controller_process = run_command(controller_command, cwd=tracker_dir)
            time.sleep(20)  # 노드 초기화 대기
            
            # # 5. 모델 학습
            # training_command = "bash -c 'source ~/.bashrc && source ./install/local_setup.bash && ros2 run utils RL_online_train'"
            # train_process = run_command(training_command, wait=True)
            
            # 학습이 완료되면, 실행된 모든 프로세스를 종료합니다.
            print("학습 반복이 완료되었습니다. 모든 프로세스를 종료합니다...")
            for proc in [yolo_process, iris_controller_process, microxrce_process, sim_process]:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"프로세스 종료 중 오류: {e}")
            
            # 다음 반복 시작 전 잠시 대기
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("사용자에 의해 자동화가 중단되었습니다.")
        for proc in [yolo_process, iris_controller_process, microxrce_process, sim_process]:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                print(f"종료 중 오류: {e}")
        print("모든 프로세스가 종료되었습니다. 자동화 스크립트를 종료합니다.")

if __name__ == "__main__":
    main()
