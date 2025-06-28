#pragma once
#include <iostream>
#include <vector>
#include <cmath>
class NeuralClass
{
public:

	NeuralClass();

	// ініціалізація ваги
	double w_1 = 0.0;
	double w_2 = 0.0;
	double w_3 = 0.0;

	// коефіціент навчання
	double nL = 0.2;

	// зміщення (bias)
	double bias = 1;

	//вхідні дані
	std::vector<std::vector<int>> inputs{};

	//таргет (очікуваний результат)
	std::vector<int> vector{};

	// прохід вперід
	double getY(double x1, double x2);
	double sigmoidFunc(double y);
	double treshholdOut(double out);

	// прохід назад
	double getError(double target, double result);
	void updateWeights(double x1, double x2, double error);

};

