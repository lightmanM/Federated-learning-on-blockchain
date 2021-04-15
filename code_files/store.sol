pragma solidity ^0.5.1;
contract store{
    // create an array of varaibles
    // set some variables in it
    // access it through python
    // update it trough python
    //uint flag = 0;
    uint[5] integerArray; 
    uint[] dynamicArray;
    int [2][5] twodarray; // stores weights of l1
    int [5][1] twodarray2; // stores weights of l2
    int [5] flag; // flag variable is size of number of addresses in the netwo
    int accuracy = 0;
    int test = -5;
    // constructor to store empty values in the array
    // function to add values to this array
    function settest(int val) public{
        test = val;
    }
    function gettest() public returns (int) {
        return test;
    }
    function addValue() public
    {
        // dummy function
        for (uint j = 0; j<4; j++)
        {
            integerArray[j] = j;
        }

    }
    // function to add data to a dynamic array
    function addDataDynamic(uint val) public {
        //dummy
        dynamicArray.push(val);
        
    }
    /* function to get the data of a dynamic array */
    function getDynamicArray() public view returns (uint[] memory){
        //dummy
        return dynamicArray;

    }
    // function to get length of dynamic array
    function getLength() public view returns (uint)
    {
        //dummy
        return dynamicArray.length;
    }
    function printVal() public view returns (uint[5] memory){
         //for (uint j = 0; j < integerArray.length; j++) {
            //print (integerArray[j]);
            //console.log(integerArray[j]);
        //}
        //dummy
        return integerArray;
    }
    /*
    function addValueTwoDarray() public
    {
        
        // for neural net here we know the size of the 2d arrays
        for (int m=0;m<twodarray.length;m++)
        {
            for (int k =0; k<twodarray[m].length;k++)
            {
                
                twodarray[m][k] = m;
                
            }
            
        }
        
    }*/
    // actual functions start from here
    function getwodarray1() public view returns(int[2][5] memory)
    {
        return twodarray;
    }
    function getwodarray2() public view returns(int[5][1] memory)
    {
        return twodarray2;
    }
    function store2darray1(uint m,uint k,int256 val) public{
        // stores the value at specified index for 2d arrays
        twodarray[m][k] = val;
    }
    function store2darray2(uint m,uint k,int val) public{
        // stores the value at specified index for 2d arrays- for 2nd weight matrix
        twodarray2[m][k] = val;
    }
    /*
    function get2darraylen() public view returns (int ){
         //for (uint j = 0; j < integerArray.length; j++) {
            //print (integerArray[j]);
            //console.log(integerArray[j]);
        //}
        return twodarray.length;
    }
    */
    function getAccuracy() public returns (int)
    {
        return accuracy;
    }
    function setAccuracy (int val) public
    {
        accuracy = val;
    }
 
}