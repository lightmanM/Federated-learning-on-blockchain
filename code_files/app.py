# for smart contract
"""

array1
array 2
accuracy
flag
"""
"""
# GET ALL ACCOUNTS AND SET UP WEB3
SET FLAG = 0 FOR ALL
GET THE ACCURACY FROM BLOCKCHAIN
SEND THE WEIGHTS TO THE SPECIFIED FILE OF THAT ADDRESS (FOR NOW - JUST CREATE MULTIPLE FILES)
#------ML FILE CODE-------------#
AFTER THE FLAG VARIABLE IS RETURNED, MOVE TO NEXT FILE





"""
# ML code
"""
Get the list of accounts available in Ganache
set the flag variable
for now - I HAVE HARD CODED A VARIABLE CALLED ADDRESS IN EACH FILE - BUT IN ACTUAL WE WILL HAVE TO CHECK

Here first call the ML weights from that specified node
# see HOW ML PROGRAM WILL BE CALLED FROM DIFFERENT NODES
if the accuracy is greater then update the ML weights in the blockchain and all other variables
# Incase of lot of data - we will have to divide the data among nodes - but that's for later now
"""

# In the ML code file:
"""
#download the weights

"""

# code starts here
import json
#from web3 import web3practice
from web3 import Web3
import app_ml as am1 # file for calling functions of first node
import app_ml2 as am2
import app_ml3 as am3

play_functions = [am1.play, am2.play, am3.play ]
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
print (web3.isConnected())
truffleFile = json.load(open('src/abis/store.json'))
abi = truffleFile['abi']
contract_address_string = truffleFile['networks']['5777']['address'] # contract address
print (contract_address_string)
contract_address = web3.toChecksumAddress(contract_address_string)
accounts = web3.eth.accounts
contract = web3.eth.contract(address = contract_address, abi = abi)
first = 1 # set this to 1 only if this is the first time this program is being run
#for i in range (len(accounts)):
for i in range(3):
    # get flag
    # get accuracy
    # send variables to all files
    # get updates
    add = accounts[i]
    print (add)
    #first = am.play(web3,add,first,i, contract)
    first = play_functions[i](web3,add,first,i, contract)
    
