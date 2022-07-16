from typing import Any
import numpy as np
import numba as nb
import string


class DataContainer:

    def __init__(self, data: Any, secret_key: str = '3BE32552183D04F8551807F8D67D3DB707C6FC3E') -> None:
        if secret_key == '3BE32552183D04F8551807F8D67D3DB707C6FC3E':
            print("You're using the default secret key.")
        self.data = data
        self.extract_len = len(data)
        self.secret_key = secret_key
        self.parameter_generator()
        if isinstance(data, str):
            self.data_type = "text"
        else:
            self.data_type = "image"

    def parameter_generator(self):
        chunks = [self.secret_key[i:i+10] for i in range(0, 40, 10)]
        r_generator = chunks[0]
        x0_generator = chunks[1]
        base_generator = chunks[2]
        iteration_generator = chunks[3]
        hex_dict = {"A": "41", "B": "52", "C": "63",
                    "D": "74", "E": "85", "F": "96"}

        r = "".join([x if x in string.digits else hex_dict[x]
                    for x in r_generator])
        small_list = "012345"
        if r[0] in small_list:
            r = '6' + r
        self.r = float("3." + r)

        x0 = "".join([x if x in string.digits else hex_dict[x]
                     for x in x0_generator])
        self.x0 = float("0." + x0)

        base = "".join([x if x in string.digits else hex_dict[x]
                       for x in base_generator])
        self.base = float(base[0] + "." + base[1:])

        self.iteration_number = int(
            iteration_generator[len(iteration_generator)//2], 16)
        if self.iteration_number == 0:
            self.iteration_number += 2


class Encryption(DataContainer):

    def __init__(self, data: Any, secret_key: str = '3BE32552183D04F8551807F8D67D3DB707C6FC3E') -> None:
        super().__init__(data, secret_key)

    def text_pre_processor(self, plain_text, base):
        data = np.array([ord(c) for c in plain_text])
        p_array = np.log(data)/np.log(base)
        n = len(data)
        if (n > 1) and (n % 2 == 0):
            rows = cols = n//2
        elif (n > 1) and (n % 2 != 0):
            rows = cols = (n+1)//2
        else:
            rows = cols = n
        k = rows * cols - n
        original_data = np.concatenate(
            (p_array, np.random.uniform(4.17, 4.80, [k])))
        return original_data.reshape((rows, cols))

    def image_pre_processor(self, image, base):
        add_array = np.ones((image.shape[0], image.shape[1]))
        new_array = np.add(image, add_array)
        p_array = np.log(new_array)/np.log(base)
        return p_array

    @staticmethod
    @nb.jit(nopython=True)
    def gen_logistic_map(dim, r, x0):
        x = [r*x0*(1-x0)]
        sequenceSize = dim * dim - 1
        for _ in range(sequenceSize):
            x.append(r*x[-1]*(1-x[-1]))
        return np.array(x).reshape(dim, dim)

    def logistic_float(self, imageMatrix, r, x0):
        dim = imageMatrix.shape
        transformationMatrix = np.array(self.gen_logistic_map(dim[0], r, x0))
        return np.bitwise_xor(imageMatrix.view(np.uint64), transformationMatrix.view(np.uint64))

    @staticmethod
    @nb.njit(parallel=True)
    def cat_mapping_fast(matrix, MAX):
        height, width = matrix.shape
        buff1 = matrix
        buff2 = np.empty((height, width))
        for _ in range(MAX):
            for y in nb.prange(height):
                for x in nb.prange(width):
                    nx = (2 * x + y) % width
                    ny = (x + y) % height
                    buff2[ny, nx] = buff1[y, x]
            buff1, buff2 = buff2, buff1
        return buff1

    def text_encryption(self, plain_text, MAX, r, x0, base):
        data = self.text_pre_processor(
            plain_text, base)
        ard_matrix = self.cat_mapping_fast(data, MAX)
        cipher_text = self.logistic_float(ard_matrix, r, x0)
        return cipher_text

    def image_encryption(self, plain_text, MAX, r, x0, base):
        data = self.image_pre_processor(
            plain_text, base)
        ard_matrix = self.cat_mapping_fast(data, MAX)
        cipher_text = self.logistic_float(ard_matrix, r, x0)
        return cipher_text

    def encrypt(self):
        if self.data_type == "text":
            cipher_text = self.text_encryption(
                self.data, self.iteration_number, self.r, self.x0, self.base)
            return cipher_text
        elif self.data_type == "image":
            cipher_text = self.image_encryption(
                self.data, self.iteration_number, self.r, self.x0, self.base
            )
            return cipher_text


class Decryption:

    def __init__(self, cipher_text: Any, data_type: str, secret_key: str = '3BE32552183D04F8551807F8D67D3DB707C6FC3E') -> None:
        if secret_key == '3BE32552183D04F8551807F8D67D3DB707C6FC3E':
            print("You're using the default secret key.")
        self.cipher_text = cipher_text
        self.secret_key = secret_key
        self.data_type = data_type
        self.parameter_generator()

        n = self.cipher_text.shape[0]
        if ((2*n-1) % 2 == 0):
            self.extract_len = 2*n - 1
        else:
            self.extract_len = 2*n

    def parameter_generator(self):
        chunks = [self.secret_key[i:i+10] for i in range(0, 40, 10)]
        r_generator = chunks[0]
        x0_generator = chunks[1]
        base_generator = chunks[2]
        iteration_generator = chunks[3]
        hex_dict = {"A": "41", "B": "52", "C": "63",
                    "D": "74", "E": "85", "F": "96"}

        r = "".join([x if x in string.digits else hex_dict[x]
                    for x in r_generator])
        small_list = "012345"
        if r[0] in small_list:
            r = '6' + r
        self.r = float("3." + r)

        x0 = "".join([x if x in string.digits else hex_dict[x]
                     for x in x0_generator])
        self.x0 = float("0." + x0)

        base = "".join([x if x in string.digits else hex_dict[x]
                       for x in base_generator])
        self.base = float(base[0] + "." + base[1:])

        self.iteration_number = int(
            iteration_generator[len(iteration_generator)//2], 16)
        if self.iteration_number == 0:
            self.iteration_number += 2

    @staticmethod
    @nb.jit(nopython=True)
    def gen_logistic_map(dim, r, x0):
        x = [r*x0*(1-x0)]
        sequenceSize = dim * dim - 1
        for _ in range(sequenceSize):
            x.append(r*x[-1]*(1-x[-1]))
        return np.array(x).reshape(dim, dim)

    def logistic_float(self, imageMatrix, r, x0):
        dim = imageMatrix.shape
        transformationMatrix = np.array(self.gen_logistic_map(dim[0], r, x0))
        return np.bitwise_xor(imageMatrix, transformationMatrix.view(np.uint64)).view(np.float64)

    @staticmethod
    @nb.njit(parallel=True)
    def reverse_cat_mapping_fast(matrix, MAX):
        height, width = matrix.shape
        buff1 = matrix
        buff2 = np.empty((height, width))
        for _ in range(MAX):
            for y in nb.prange(height):
                for x in range(width):
                    nx = (x - y) % width
                    ny = (-x + 2*y) % height
                    buff2[ny, nx] = buff1[y, x]
            buff1, buff2 = buff2, buff1
        return buff1

    def text_post_processor(self, data, extract_len, base):
        data = data.reshape(-1)
        clean_array = [round(base**asci) for asci in data[:extract_len]]
        list_string = [chr(i) for i in clean_array]
        decrypted_msg = ''.join(list_string)
        return decrypted_msg

    def image_post_processor(self, data, base):
        subtract_array = np.ones((data.shape[0], data.shape[1]))
        data_shape = data.shape
        data = data.reshape(-1)
        data = np.array([round(base**entry) for entry in data[:]], dtype=np.int64)
        data = data.reshape(data_shape)
        data = np.subtract(data, subtract_array)
        return data

    def decrypt(self):
        rehe_matrix = self.logistic_float(self.cipher_text, self.r, self.x0)
        re_matrix = self.reverse_cat_mapping_fast(rehe_matrix, self.iteration_number)

        if self.data_type == "text":
            plain_text = self.text_post_processor(
                re_matrix, self.extract_len, self.base)
            return plain_text
        elif self.data_type == "image":
            plain_text = self.image_post_processor(re_matrix, self.base)
            return plain_text