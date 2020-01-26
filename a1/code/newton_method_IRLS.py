from logistic_regression_irls import *
import matplotlib.pyplot as plt

def newton_method(X, y, iterations, lamb):
    w = np.random.normal(0, 0.1, (n_features,))
    loss_curves = {'train': [], 'test': []}
    accuracy_curves = {'train': [], 'test': []}

    for t in range(iterations):
        print("iteration: ", t)

        # CALCULATE GRADIENT & HESSIAN
        grad_L = grad_Loss(X, w, y, lamb)
        H = hessian(X, w, lamb)

        # UPDATE STEP
        # we calculate pseudo-inverse becuase overflow causes H to converge to 0
        w = w - np.linalg.pinv(H) @ grad_L

        # CALCULATE LOSS & ACCURACY
        loss_train = loss(X_train, w, y_train, lamb)
        loss_curves['train'].append(loss_train)
        loss_test = loss(X_test, w, y_test, lamb)
        loss_curves['test'].append(loss_test)

        acc_train = accuracy(X_train, w, y_train)
        accuracy_curves['train'].append(acc_train)
        acc_test = accuracy(X_test, w, y_test)
        accuracy_curves['test'].append(acc_test)

    return w, loss_curves, accuracy_curves


def main():
    w, loss, accuracy = newton_method(X_train, y_train, iterations=7, lamb=1)

    print("-------------------------------------------------")
    print("Training Data:")
    print("Loss: ", loss['train'][-1])
    print("Accuracy: ", accuracy['train'][-1])
    print("-------------------------------------------------")
    print("Test Data:")
    print("Loss: ", loss['test'][-1])
    print("Accuracy: ", accuracy['test'][-1])
    print("-------------------------------------------------")

    # PLOT LOSS CURVE
    plt.plot(loss['train'], color='blue', label='training data')
    plt.plot(loss['test'], color='green', label='test data')
    plt.legend()
    plt.title('Loss Curves')
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.show()

    # PLOT ACCURACY CURVE
    plt.plot(accuracy['train'], color='blue', label='training data')
    plt.plot(accuracy['test'], color='green', label='test data')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    main()
