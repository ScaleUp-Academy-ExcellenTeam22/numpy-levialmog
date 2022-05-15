import numpy as np
import matplotlib.pyplot as plt


# ----------q1----------
def question1():
    """
    The function creates a vector with values from 0 to 20 and change the sign of the numbers in the range from 9 to 15.
    :return: The vector after the change.
    """
    vector = np.array(range(21))
    vector[9:16] *= -1
    return vector


# ----------q2----------
def question2():
    """
    The function creates a vector of length 10 with values evenly distributed between 5 and 50.
    :return: The vector that created.
    """
    return np.linspace(5, 50, 10)


# ----------q3----------
def question3(matrix):
    """
    The function finds number of rows and columns of a given matrix.
    :param matrix: The given matrix.
    :return: The matrix after the change.
    """
    return np.array(matrix).shape


# ----------q4----------
def question4():
    """
    The function creates a 10x10 matrix, in which the elements on the borders will be equal to 1, and inside 0.
    :return: The matrix that created.
    """
    matrix = np.ones((10, 10))
    matrix[1:-1, 1:-1] = 0
    return matrix


# ----------q5----------
def question5(matrix, vector):
    """
    The function adds the vector to each row of a given matrix.
    :param matrix: The given matrix.
    :param vector: The vector that adds.
    :return: The matrix after change.
    """
    matrix_after_add = np.empty_like(matrix)
    for row in range(len(matrix)):
        matrix_after_add[row, :] = matrix[row, :] + vector

    return matrix_after_add


# ----------q6----------
def question6(x, y):
    """
    The function computes the x and y coordinates for points on a sine curve and plot the points using matplotlib.
    :param x: coordinate x.
    :param y: coordinate y.
    """
    x = np.arange(x, y * np.pi, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()


# ----------q7----------
def question7():
    """
    The function creates a 4x4 array with random values, after creates a new array from the said array swapping first
    and last rows.
    :return: The matrix after change.
    """
    matrix = np.random.rand(4, 4)
    return matrix[::-1]


# ----------q8----------
def question8(matrix, number):
    """
    The function replaces all numbers in a given array which is equal, less and greater to a given number.
    The function prints the results.
    :param matrix: The given matrix.
    :param number: The number with the exchange is made.
    """
    number_to_replace = 0

    print(np.where(matrix == number, number_to_replace, matrix))
    print(np.where(matrix < number, number_to_replace, matrix))
    print(np.where(matrix > number, number_to_replace, matrix))


# ----------q9----------
def question9(matrix1, matrix2):
    """
    The function multiplies two given arrays of same size element-by-element.
    :param matrix1: The first given matrix.
    :param matrix2: The second given matrix.
    :return: The multiplied matrix.
    """
    return np.multiply(matrix1, matrix2)


# ----------q10----------
def question10(matrix):
    """
    The function sorts an along the first, last axis of an array.
    :param matrix: Te given matrix.
    """
    first_axis_sorted = np.sort(matrix, axis=0)
    print(first_axis_sorted)
    last_axis_sorted = np.sort(first_axis_sorted, axis=1)
    print(last_axis_sorted)


# ----------q11----------
def question11():
    """
    The function creates a 3-D array with ones on a diagonal and zeros elsewhere.
    :return: The 3-D array.
    """
    return np.eye(3)


# ----------q12----------
def question12(matrix):
    """
    The function removes single-dimensional entries from a specified shape.
    :param matrix: The given matrix.
    :return: The matrix after change.
    """
    return np.squeeze(matrix)


# ----------q13----------
def question13(*arrays):
    """
    The function converts (in sequence depth wise (along third axis)) two 1-D arrays into a 2-D array.
    :param arrays: The given arrays.
    :return: the array after change.
    """
    return np.dstack(arrays)


# ----------q14----------
def question14(vector, matrix):
    """
    The function combines a one and a two dimensional array together and display their elements.
    The function prints the results.
    :param vector: The given vector.
    :param matrix: the given matrix.
    """
    for index, element in np.nditer([vector, matrix]):
        print("%d:%d" % (index, element),)


# ----------q15----------
def question15():
    """
    The function creates a 3-dimension array with shape (300,400,5) and set to a variable. Fill the array elements
    with values using unsigned integer (0 to 255).
    :return: The 3-dimension array
    """
    return np.random.randint(low=0, high=256, size=(300, 400, 5), dtype=np.uint8)


# ----------q16----------
def question16(students_id, students_height):
    """
    The function sorts the student id with increasing height of the students from given students id and height.
    The function prints the results.
    :param students_id: The given students id.
    :param students_height: The given students height.
    """
    for n in np.lexsort((students_id, students_height)):
        print(students_id[n], students_height[n])


# ----------q17----------
def question17(array):
    """
    The function computes the median of flattened given array.
    :param array: The given array.
    :return: The median.
    """
    return np.median(array)


# ----------q18----------
def question18(date1, date2):
    """
    The function counts the number of days of specific month.
    :param date1: The first given date.
    :param date2: The second given date.
    :return: The number of days.
    """
    return np.datetime64(date1) - np.datetime64(date2)


# ----------q19----------
def question19():
    """
    The function creates the French flag using NumPy and the convert it to an image.
    """
    color_flag = np.array([(0, 0, 255), (255, 255, 255), (255, 0, 0)])
    plt.show(color_flag)


if __name__ == "__main__":
    print(list(question1()))
    print(list(question2()))
    print(question3([[0, 0, 0], [0, 0, 0]]))
    print(question4())
    print(question5(np.array([[0, 0, 0], [0, 0, 0]]), np.array([1, 2, 3])))
    question6(0, 5)
    print(question7())
    question8(np.array([[1, 2, 3], [4, 5, 6]]), 3)
    print(question9(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]])))
    question10(np.array([[4, 6], [2, 1]]))
    print(question11())
    print(question12([[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]]))
    print(question13(np.array((10, 20, 30)), np.array((40, 50, 60))))
    question14(np.array([1, 2, 3]), np.array([[1, 2, 3], [4, 5, 6]]))
    print(question15())
    question16(np.array([123, 456, 789]), np.array([1.70, 1.85, 1.65]))
    print(question17([[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]]))
    print(question18('2016-03-01', '2016-02-01'))
    question19()

