#ifndef __READER__

#include <string>
#include <fstream>//ifstream,output
#include <sstream>//istringstream
#include <iostream>//cin,cout,output
#include <stdlib.h>//strtof

typedef struct InputParameters
{
	int Lx,Ly,Lz;
	double magneticField;
	int MonteCarloSteps;
	double highestTemperature;
} InputParameters;

typedef struct Reader
{
	InputParameters inputs;

	Reader(std::string fileName)
	{
		int parametersRead = 0;
		inputs.Lx = inputs.Ly = inputs.Lz = 1;
		std::ifstream fileStream(fileName.c_str());
		if(not fileStream)
		{
			std::cout << "Input file not found. Exiting." << std::endl;
		    exit(EXIT_FAILURE);
		}
		std::string line;
		while(std::getline(fileStream, line))
		{
			std::istringstream stringStream(line);
			std::string parameter,valueString;
			double value;
			//Parameter name.
			std::getline(stringStream, parameter,',');
			//Parameter value.
			std::getline(stringStream, valueString,',');
			value = strtof(valueString.c_str(),NULL);
			//Interpretation.
			if(parameter=="Lx")
			{
				inputs.Lx = value;
				++parametersRead;
				std::cout << "Lx Read: " << inputs.Lx << std::endl;
			}
			else if(parameter=="Ly")
			{
				inputs.Ly = value;
				++parametersRead;
				std::cout << "Ly Read: " << inputs.Ly << std::endl;
			}
			else if(parameter=="Lz")
			{
				inputs.Lz = value;
				++parametersRead;
				std::cout << "Lz Read: " << inputs.Lz << std::endl;
			}
			else if(parameter=="magneticField")
			{
				inputs.magneticField = value;
				++parametersRead;
				std::cout << "magneticField Read: " << inputs.magneticField << std::endl;
			}
			else if(parameter=="MonteCarloSteps")
			{
				inputs.MonteCarloSteps = value;
				++parametersRead;
				std::cout << "MonteCarloSteps Read: " << inputs.MonteCarloSteps << std::endl;
			}
			else if(parameter=="highestTemperature")
			{
				inputs.highestTemperature = value;
				++parametersRead;
				std::cout << "highestTemperature Read: " << inputs.highestTemperature << std::endl;
			}
			else
			{
				std::cout << "Parameter '" << parameter << "' was not correctly identified and assigned. Please correct any errors in the input file." << std::endl;
			}
		}
		if(parametersRead < 6)
		{
			std::cout << "Not enough parameters (only " << parametersRead << ") were read from the input file." << std::endl;
	    	std::cout << "Exiting." << std::endl;
	    	exit(EXIT_FAILURE);
		}
	}


} Reader;


#endif