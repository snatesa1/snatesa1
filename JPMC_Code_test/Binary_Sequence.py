import numpy as np

def convert_to_binary(num):
    try:
        return np.binary_repr(num)
    except TypeError:
        return "Invalid input. Please enter a valid integer."

def find_longest_sequence(binary_str):
    max_start = 0
    max_length = 0
    current_start = 0
    current_length = 0

    for i, value in enumerate(binary_str):
        if value == '1':
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                max_start = current_start  
        else:
            current_start = i + 1
            current_length = 0

    # Check if no continuous sequence was found
    if max_length == 0:
        return None  
    else:
        return max_start  # Return max_start, not max_length


def main():
    number = int(input("Enter the number:"))
    binary_str = convert_to_binary(number)
    longest_start = find_longest_sequence(binary_str)

    print(f"Input: {number}")
    print(f"Binary format: {binary_str}")
    print(f"Starting position of the longest 1's sequence: {longest_start}")

if __name__ == "__main__":
    main()
