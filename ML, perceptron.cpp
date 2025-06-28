#include "NeuralClass.h"
int main() {
    NeuralClass nn;

    // Ініціалізуємо випадкові ваги
    srand(time(0));
    nn.w_1 = ((double)rand() / RAND_MAX) - 0.5;
    nn.w_2 = ((double)rand() / RAND_MAX) - 0.5;
    nn.w_3 = ((double)rand() / RAND_MAX) - 0.5;

    std::vector<std::vector<int>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<int> targets = { 0, 0, 0, 1 }; // AND

    int epoch = 0;
    while (true) {
        bool allcorrect = true;
        for (int i = 0; i < inputs.size(); ++i) {
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double y = nn.getY(x1, x2);
            double out = nn.sigmoidFunc(y);
            double result = nn.treshholdOut(out);
            double error = nn.getError(targets[i], result);
            nn.updateWeights(x1, x2, error);

            if (result != targets[i]) {
                allcorrect = false;
            }
        }
        epoch++;
        if (allcorrect || epoch >= 10000) break;
    }


    std::cout << "Training finished!" << std::endl;

    for (int i = 0; i < inputs.size(); ++i) {
        double x1 = inputs[i][0];
        double x2 = inputs[i][1];

        double y = nn.getY(x1, x2);
        double out = nn.sigmoidFunc(y);
        double result = nn.treshholdOut(out);

        std::cout << x1 << " " << x2 << " -> " << result << std::endl;
    }
    std::cout << "Training finished in " << epoch << " epochs!" << std::endl;
    return 0;
}