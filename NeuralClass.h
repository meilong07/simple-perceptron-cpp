#pragma once
#include <iostream>
#include <vector>
#include <cmath>
class NeuralClass
{
public:

	NeuralClass();

	// ����������� ����
	double w_1 = 0.0;
	double w_2 = 0.0;
	double w_3 = 0.0;

	// ���������� ��������
	double nL = 0.2;

	// ������� (bias)
	double bias = 1;

	//����� ���
	std::vector<std::vector<int>> inputs{};

	//������ (���������� ���������)
	std::vector<int> vector{};

	// ������ �����
	double getY(double x1, double x2);
	double sigmoidFunc(double y);
	double treshholdOut(double out);

	// ������ �����
	double getError(double target, double result);
	void updateWeights(double x1, double x2, double error);

};

