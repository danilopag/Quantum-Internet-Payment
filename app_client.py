import json
import math
import random
import binascii
import hashlib
import hmac

from dataclasses import dataclass
from typing import Optional

from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk import EPRSocket
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import NetQASMConnection, Socket

logger = get_netqasm_logger()

def receive_states(conn, epr_socket, socket, target, n):
    bit_flips = [None for _ in range(n)]
    basis_flips = [random.randint(0, 1) for _ in range(n)]

    for i in range(n):
        q = epr_socket.recv_keep(1)[0]
        if basis_flips[i]:
            q.H()
        m = q.measure()
        conn.flush()
        bit_flips[i] = int(m)

    return bit_flips, basis_flips

def receive_states(conn, epr_socket, socket, target, n, basis_flips):
    bit_flips = [None for _ in range(n)]

    for i in range(n):
        q = epr_socket.recv_keep(1)[0]
        if basis_flips[i]:
            q.H()
        m = q.measure()
        conn.flush()
        bit_flips[i] = int(m)

    return bit_flips, basis_flips


def compute_hmac(key, message):
    # Usa SHA-256 come funzione hash di base
    hash_algorithm = hashlib.sha256

    # Calcola l'HMAC utilizzando la chiave e il messaggio
    hmac_result = hmac.new(key, message, hash_algorithm)

    # Converte l'HMAC in una stringa binaria
    hmac_binary = binascii.hexlify(hmac_result.digest())
    return hmac_binary


def main(app_config=None, num_bits=128, merchant_list=None, id_client_bank=None):
    print("\n ------------------------------------- \n")
    print("Ciao sono Client, apro il Socket con TTP")
    print(f"Il mio codice cliente è {id_client_bank}")
    print("\n ------------------------------------- \n")
    # Socket for classical communication
    socket = Socket("client", "ttp", log_config=app_config.log_config)
    # Socket for EPR generation
    epr_socket = EPRSocket("ttp")
    """Scelta del commerciante da contattare"""
    print("Scegli il commerciante da cui effettuare l'acquisto")
    for i, merc in enumerate(merchant_list, start=1):
    	print(f"{i}. {merc}")
    scelta = input("Inserisci il numero del commerciante da contattare: ");
    try:
    	choice = int(scelta)
    	socket2 = Socket("client","merchant", log_config=app_config.log_config)
    	if 1 <= choice <= len(merchant_list):
    		merchant_choice = merchant_list[choice - 1]
    		print(f"Hai scelto di effettuare la transazione presso: {merchant_choice}")
    	else:
    		print("Scelta non valida. Inserisci un numero valido")
    except ValueError:
    	print("Input non valido. Inserisci un numero.") 
    	
    """Costruzione del HMAC"""
    key_string = str(id_client_bank)
    key = key_string.encode()
    message = merchant_list[choice - 1].encode() 
    mac = compute_hmac(key, message)
    print(f'Message: {message.decode()}')
    print(f'HMAC (binary): {mac}')
    binary_string = bin(int(mac, 16))[2:]
    bit_list = [int(bit) for bit in binary_string]
    bit_list = bit_list[:128]
    
    bases_token = [int(b) for b in bit_list]
    print(f"Base misurazione token di pagamento{bases_token}")
    
    """Socket nel quale il Client riceve il token di pagamento"""
    client = NetQASMConnection(
        app_name=app_config.app_name,
        log_config=app_config.log_config,
        epr_sockets=[epr_socket],
    )
    with client:
        bit_flips, basis_flips = receive_states(
            client, epr_socket, socket, "ttp", num_bits, bases_token
        )
    outcomes = [int(b) for b in bit_flips]
    basis = [int(b) for b in basis_flips]
    print(f"Client raw key: {outcomes}")
    print("Mando al commerciante il token di pagamento")
    print("\n ------------------------------------- \n")
    msgMerchant = StructuredMessage(header="InfoToken", payload=[outcomes,id_client_bank]) 
    """INVIO DELLA PROPRIA BASE"""
    socket2.send_structured(msgMerchant)
    return{"raw_key": outcomes,}


if __name__ == "__main__":
    main()
