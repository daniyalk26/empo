"""# Author: Muzammil Nasim
# Copyright : Muzammil Nasim ⓒ 2023 - All rights reserved
# This file contains sensitive code not to be shared by unauthorized individuals.`
"""
# pylint: disable=invalid-name,E1120,C0301,C0305,C0114,W0603,C0116,W0621,C0103
# pylint: disable=W0718,C0115,R0205,R1732,W0612

import base64
import hashlib
import os
import struct
from tempfile import SpooledTemporaryFile
import Crypto
from Crypto import Random, PublicKey
from Crypto.Cipher import AES
from Crypto.Hash import SHA512
from Crypto.Protocol.KDF import PBKDF2
from Crypto.PublicKey import RSA
from werkzeug.datastructures import FileStorage

# samples at the bottom of the file.
s1 = '^^442&THJVEpk3tQ8!dQ0^Mm$!5uqNA6^^'
s2 = '^^bC1$onmcWGw$Ir3mYnK9oJVHpX80*^^^'
salt = b'3sQsTcJ**97wrWXFMx@5p^5^13ejK3a*'
db_psk = file_psk = None
psks = {}


def get_new_file_key():
    rsa = RSA.generate(2048)
    public = rsa.publickey()
    key = base64.b64encode(str(public.n).encode('utf-8'))
    ret_key = KDF2(key, s1, s2)
    return ret_key


def get_db_psk():
    # generated once and used as key
    return get_psk_by_name('db_psk.key')


def get_file_psk():
    # generated once and used as kek
    return get_psk_by_name('blob_psk.key')


def get_psk_by_name(name):
    # generated once and used as kek
    global psks
    psk = psks.get(name)
    if not psk:
        key_dir = os.getenv('KEYS_DIR', 'keys')
        data = read_key(f"{key_dir}/{name}")
        rsa = Crypto.PublicKey.RSA.import_key(data)
        public = rsa.publickey()
        key = base64.b64encode(str(public.n).encode('utf-8'))
        psk = KDF2(key, s1, s2)
        psks[name] = psk
    return psk


def read_key(path):
    with open(path, 'rb') as key_file:
        key = key_file.read()
        return key


def XOR_hex(args):
    res = ""
    i = len(args[0]) - 1
    while i >= 0:
        j = 1
        temp = int(args[0][i], 16)
        while j < len(args):
            temp ^= int(args[j][i], 16)
            j += 1
        res = format(temp, 'x') + res
        i -= 1
    return res


def KDF2(psk, s1, s2):
    decoded = base64.b64decode(psk).decode('utf-8')
    m1 = sha256_hash(s1)
    m2 = sha256_hash(s2)
    m3 = sha256_hash(decoded)
    temp = XOR_hex([m1, m2, m3])
    k = temp.encode('utf-8')
    keys = PBKDF2(k, salt[16:], 64, count=600000, hmac_hash_module=SHA512)
    return keys[32:]


def sha256_hash(message):
    try:
        message_bytes = message.encode('utf-8')
        readable_hash = hashlib.sha256(message_bytes).hexdigest()
        return readable_hash
    # pylint: disable=W0612
    except Exception as e:
        return None


class AESCipher(object):
    def __init__(self, key, auth_key=None):
        self.bs = AES.block_size
        self.key = key
        self.auth_key = auth_key

    def encrypt(self, raw):
        if isinstance(raw, str):
            raw = raw.encode('UTF-8')
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, encrypted, auth=False):
        joinedData = base64.b64decode(encrypted)
        iv = joinedData[:16]  # Determine IV from concatenated data
        msg = joinedData[16:]  # Determine ciphertext from concatenated data
        decrypted = self.unpadPkcs7(
            self.decrypt_internal(msg, iv, auth))  # Decrypt and remove PKCS7-padding manually
        return decrypted

    # creates an in memory temp file.
    def decrypt_file(self, in_file, out_filename):
        in_f = None
        in_opened = False
        try:
            if isinstance(in_file, str):
                in_f = open(in_file, 'rb')
                in_opened = True
            else:
                in_f = in_file
            out_f = SpooledTemporaryFile(mode='wb+')
            ret_val = self.decrypt_file_internal(in_f, out_f)
            if ret_val is None:
                return ret_val
            ret_file = FileStorage(stream=out_f, filename=out_filename, name=out_filename)
            return ret_file
        except Exception as e:
            print(e)
            return None
        finally:
            if in_opened:
                in_f.close()

    def encrypt_file(self, in_file, outfile_name):
        in_f = None
        in_opened = False
        try:
            if isinstance(in_file, str):
                in_f = open(in_file, 'rb')
                in_opened = True
            else:
                in_f = in_file
            out_f = SpooledTemporaryFile(mode='wb+')
            if not self.encrypt_file_internal(in_f, out_f):
                return None
            out_f.seek(0, 0)
            return FileStorage(stream=out_f, name=outfile_name, filename=outfile_name)
        except Exception as e:
            print(e)
            return None
        finally:
            if in_opened:
                in_f.close()

    # The functions below are meant to be internally used by above functions
    def decrypt_file_internal(self, in_file, out_file):
        try:
            while True:
                chunk_length = File.read_int(in_file)
                chunk = in_file.read(chunk_length)
                if not chunk:
                    break
                joinedData = base64.b64decode(chunk)
                iv = joinedData[:16]  # Determine IV from concatenated data
                msg = joinedData[16:]
                decrypted = self.unpadPkcs7(self.decrypt_internal(msg, iv, False), False)
                out_file.write(decrypted)
                out_file.flush()
            out_file.seek(0, 0)
            return FileStorage(stream=out_file)
        except Exception as e:
            print(e)
            return None

    def encrypt_file_internal(self, in_file, out_file):
        try:
            while True:
                chunk = in_file.read(1024 * self.bs)
                if not chunk:
                    break

                encrypted = self.encrypt(chunk)
                File.write_int(len(encrypted), out_file)
                out_file.write(encrypted)
            return out_file
        except Exception as e:
            print(e)
            return None

    def decrypt_internal(self, encrypted, iv, auth=False):
        key = None
        if auth:
            key = self.auth_key
        else:
            key = self.key  # Interpret key as Base64 encoded
        aes = AES.new(key, AES.MODE_CBC, iv)  # Use CBC-mode
        decrypted = aes.decrypt(encrypted)  # decode
        return decrypted

    def unpadPkcs7(self, data, decode=True):
        if decode:
            try:
                data = data[:-data[-1]].decode('utf-8')
            # pylint: disable=W0612
            except Exception as e:
                data = data[:-data[-1]]
        else:
            data = data[:-data[-1]]
        return data

    def _pad(self, s):
        if isinstance(s, bytes):
            return s + bytes((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs),
                             'ascii')
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)


class File:
    def __init__(self):
        pass

    @staticmethod
    def read_int(file):
        temp = file.read(4)
        # byt = struct.unpack('i', temp)
        val = int.from_bytes(temp, 'big')
        return val

    @staticmethod
    def write_int(val, file):
        data = struct.pack('>i', val, )
        file.write(data)
