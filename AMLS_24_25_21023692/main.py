# Python Script for main
import os


def run_script(script_path):
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, globals())


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    TaskACNN_path = os.path.join(base_dir, 'A/TaskACNN.py')
    TaskASVM_path = os.path.join(base_dir, 'A/TaskASVM.py')
    TaskBCNN_path = os.path.join(base_dir, 'B/TaskBCNN.py')
    # TaskBSVM_path = os.path.join(base_dir, 'B/TaskBSVM.py')
    run_script(TaskACNN_path)
    run_script(TaskASVM_path)
    run_script(TaskBCNN_path)
    # run_script(TaskBSVM_path)
