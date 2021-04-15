import jsonfrom web3 import web3practice


ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

abi = '' #for now get this from remix after compiling
bytecode = ''#for now get this from remix after compiling

Greeter = web3.eth.contract(abi = abi, bytecode = bytecode) #calling the constructor
tx_hash = Greeter.constructor().transact()
tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

contract = web3.eth.contract(
    address = tx_receipt.contractAddress,
    abi = abi
)

print(contract.functions.greet().call())
tx_hash = contract.functions.setGreeting("Hellooooooooo!!").transact()
tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
print (contract.functions.greet().call())

# to compile with solidity compiler instead of using remix, we need to download a solidity compiler

# check web3 docs
