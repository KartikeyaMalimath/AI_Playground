from tic_tac_toe_ai.tic_tac_toe_ai import initialize_game
import tensorflow as tf


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        print("CUDA is enabled. Available GPU devices:")
        for device in physical_devices:
            print(device)
    else:
        print("CUDA is not enabled. Only CPU available.")
    initialize_game()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
