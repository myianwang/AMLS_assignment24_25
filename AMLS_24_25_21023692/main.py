# Python Script for main
import os


def run_script(script_path):
    """
    Run the python script at the given path.
    :param script_path: path to the python script
    """
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, globals())


if __name__ == "__main__":
    """
    Run the main scripts for Task A and Task B.
    """

    # Get the base directory
    base_dir = os.path.dirname(__file__)

    # Task A and Task B scripts
    TaskACNN_path = os.path.join(base_dir, 'A/TaskACNN.py')
    TaskASVM_path = os.path.join(base_dir, 'A/TaskASVM.py')
    TaskBCNN_path = os.path.join(base_dir, 'B/TaskBCNN.py')
    TaskBSVM_path = os.path.join(base_dir, 'B/TaskBSVM.py')

    # Run the scripts
    run_script(TaskACNN_path)
    run_script(TaskASVM_path)
    run_script(TaskBCNN_path)
    run_script(TaskBSVM_path)