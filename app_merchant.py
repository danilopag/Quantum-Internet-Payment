import json
import math
import binascii
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional
from netqasm.logging.glob import get_netqasm_logger
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import NetQASMConnection, Socket

logger = get_netqasm_logger()

def compute_hmac(key, message):
    # Usa SHA-256 come funzione hash di base
    hash_algorithm = hashlib.sha256

    # Calcola l'HMAC utilizzando la chiave e il messaggio
    hmac_result = hmac.new(key, message, hash_algorithm)

    # Converte l'HMAC in una stringa binaria
    hmac_binary = binascii.hexlify(hmac_result.digest())
    return hmac_binary


def main(app_config=None, num_bits=128, id_merchant=None):
    print("\n ------------------------------------- \n")
    print("Ciao sono Merchant, apro il Socket con il Cliente")
    # Socket for classical communication
    merchant = NetQASMConnection(
        app_name=app_config.app_name,
        log_config=app_config.log_config,
    )
    socket = Socket("merchant", "client", log_config=app_config.log_config)
    socket2 = Socket("merchant", "ttp", log_config=app_config.log_config)
    criptogram = socket.recv_structured().payload
    criptogram.append(id_merchant) 
    print(f"Client criptogram: {criptogram}")
    print(f"Mando il messaggio completo ricevuto dal Client per verificare la correttezza")
    print("\n ------------------------------------- \n")
    msgTTP = StructuredMessage(header="InfoToken", payload=criptogram) 
    """INVIO DEL TOKEN di Pagamento al TTP"""
    socket2.send_structured(msgTTP)
    msgVerifica = socket2.recv_structured().payload
    print(msgVerifica[0])
    
if __name__ == "__main__":
    main()
