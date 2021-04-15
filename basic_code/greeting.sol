pragma solidity ^0.4.21;
contract Greeter{
string public greeting;
function Greeter() public{
    greeting = 'Hello';
}
function setGreeting(string g) public {

    greeting = g;
}
function greet() view public returns (string){
    return greeting;
}
}
