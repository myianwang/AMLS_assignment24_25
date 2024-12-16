# Python Script for main
import os


def run_script(script_path):
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, globals())


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    task_a_path = os.path.join(base_dir, 'A/TaskA.py')
    task_b_path = os.path.join(base_dir, 'B/TaskB.py')
    run_script(task_a_path)
    print("-------------------")
    run_script(task_b_path)
