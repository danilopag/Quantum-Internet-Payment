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
def distribute_states(conn, epr_socket, socket, target, n):
    bit_flips = [None for _ in range(n)]
    basis_flips = [random.randint(0, 1) for _ in range(n)]

    for i in range(n):
        q = epr_socket.create_keep(1)[0]
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


def main(app_config=None, num_bits=128, id_clients_bank=None):
    num_test_bits = max(num_bits // 4, 1)
    print("\n ------------------------------------- \n")
    print("Ciao sono TTP! Apro il mio Socket per comunicare con Client")
    print("Chiave da N bit ", num_bits)
    print("Lista clienti banca:")
    for i in id_clients_bank:
    	print(f"{id_clients_bank}")
    # Socket for classical communication
    socket = Socket("ttp", "client", log_config=app_config.log_config)
    socket2 = Socket("ttp", "merchant", log_config=app_config.log_config)
    # Socket for EPR generation
    epr_socket = EPRSocket("client")

    ttp = NetQASMConnection(
        app_name=app_config.app_name,
        log_config=app_config.log_config,
        epr_sockets=[epr_socket],
    )
    with ttp:
        bit_flips, basis_flips = distribute_states(
            ttp, epr_socket, socket, "client", num_bits
        )

    outcomes = [int(b) for b in bit_flips]
    theta = [int(b) for b in basis_flips]
    
    print("\n ------------------------------------- \n")
    print("Risultati chiave TTP: {outcomes}")
    print("Risultati base TTP: {theta}")
	
    """RICEVO I DATI DA UN IPOTETICO MERCANTE"""
    remote_token = socket2.recv_structured().payload
    key_token = remote_token[0]
    print(f"TTP ho ricevuto il criptogramma: {key_token}")
   
    """Costruzione del HMAC"""
    key_string = str(remote_token[1])
    key = key_string.encode()
    message = remote_token[2].encode()
    mac = compute_hmac(key, message)
    print(f'Message: {message.decode()}')
    print(f'HMAC (binary): {mac}')
    binary_string = bin(int(mac, 16))[2:]
    bit_list = [int(bit) for bit in binary_string]
    bit_list = bit_list[:128]
    
    bases_token = [int(b) for b in bit_list]
    print(f"Base misurazione token di pagamento{bases_token}")
    print("Verifica finale della validità del token. Tolleranza 75%")
    tollerance = 0
    for i in range(len(theta)):
    	if theta[i] == bases_token[i] and outcomes[i] == key_token[i]:
    		tollerance = tollerance + 1
    if tollerance > 64:
    	print("\n ------------------------------------- \n")
    	print("Token di pagamento accettato")
    	msgMerchant = StructuredMessage(header="OutcomeToken", payload=["Pagamento approvato"])
    	socket2.send_structured(msgMerchant)
    else:
    	print("\n ------------------------------------- \n")
    	print("Token di pagamento non accettato")
    	print(f"outcomes: {outcomes} theta: {theta}")
    	print(f"key_token: {key_token} bases_token:{bases_token}")
    	msgMerchant = StructuredMessage(header="OutcomeToken", payload=["Pagamento non approvato"])
    	socket2.send_structured(msgMerchant)
    return {"raw_key" : outcomes,}


if __name__ == "__main__":
    main()
