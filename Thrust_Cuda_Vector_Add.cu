#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include<bits/stdc++.h>

int main(int argc, char *argv[]) {
	float *hostInput1 = NULL;
	float *hostInput2 = NULL;
	//float *hostOutput = NULL;
	int inputLength;

	/* parse the input arguments */
	//@@ Insert code here
	std::ifstream myIn0File;
	myIn0File.open(argv[2]);	// Reading data from input0.raw File
	myIn0File>>inputLength;

	std::ifstream myIn1File;
	myIn1File.open(argv[3]);	// Reading data from input1.raw File
	myIn1File>>inputLength;

	// Import host input data
	//@@ Read data from the raw files here
	//@@ Insert code here
	hostInput1 = (float*)malloc(inputLength*sizeof(float));
	hostInput2 = (float*)malloc(inputLength*sizeof(float));
	int i=0;
	while (!myIn0File.eof()) {
		myIn0File >> hostInput1[i++];
	}
	i=0;
	while (!myIn1File.eof()) {
		myIn1File >> hostInput2[i++];
	}
	myIn0File.close();
	myIn1File.close();

	thrust::host_vector<float> host_input1(inputLength);
	thrust::host_vector<float> host_input2(inputLength);
	for(int j=0;j<inputLength;j++){
		host_input1[j]=hostInput1[j];
		host_input2[j]=hostInput2[j];
	}

	// Declare and allocate host output
	//@@ Insert code here

	thrust::host_vector<float> host_output(inputLength);

	// Declare and allocate thrust device input and output vectors
	//@@ Insert code here
	thrust::device_vector<float> device_input1(inputLength);
	thrust::device_vector<float> device_input2(inputLength);
	thrust::device_vector<float> device_output(inputLength);

	// Copy to device
	//@@ Insert code here

	device_input1=host_input1;
	device_input2=host_input2;

	// Execute vector addition
	//@@ Insert Code here

	thrust::transform(device_input1.begin(), device_input1.end(), 	device_input2.begin(), device_output.begin(), thrust::plus<float>());
	/////////////////////////////////////////////////////////

	// Copy data back to host
	//@@ Insert code here

	host_output=device_output;

	// Comparing Expected output with program output
	// and Storing data to output.raw File

	std::ifstream expectedFile;
	expectedFile.open(argv[1]);	// Reading data from expected.raw File
	expectedFile>>inputLength;

	std::ofstream outfile (argv[4]);	  //Creating Output.raw File 
	if (outfile.is_open())
	{
		float EPSILON=0.0001;
		float expVar;
		int i=0;
		outfile <<  inputLength<< "\n";
		for(int j=0;j<inputLength;j++){
			expectedFile>>expVar;
		//Comparing actual output with expected output
		if(fabs((double)(device_output[j]-expVar)) EPSILON)
			outfile<< std::fixed << std::setprecision(2) 					<<host_output[j] << "\n";
		else{
			i=1;break;
		}
		}
		outfile.close();
		if(i==1)
			std::cout<<"\nDATA Mismatch!!!\n\n";
		else
		std::cout<<"\nBoth Expected output vector and Program Output 			Vector are same\n\n";

		//Printing Program output
		for(int j=0;j<inputLength;j++){
			std::cout.precision(2);
			std::cout << "D[" << j << "] = " << std::fixed << 					host_output[j] << "\n";
		}
	}
	free(hostInput1);
	free(hostInput2);
	//free(hostOutput);
	return 0;
}
