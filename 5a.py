import numpy as np
def perceptron(weight,bias,x):
    model=np.add(np.dot(x,weight),bias)
    logit=1/(1+np.exp(-model))
    return np.round(logit)
def compute(logictype,weightdict,dataset):
    weights=np.array([weightdict[logictype][w] for w in weightdict[logictype].keys()])
    output=np.array([perceptron(weights,weightdict['bias'][logictype],val) for val in dataset])
    print("\n", logictype.upper())
    return logictype, output
def main():
    logic = {
        'logic_and': {'w0': -0.1, 'w1': 0.2, 'w2': 0.2},
        'logic_nand': {'w0': 0.6, 'w1': -0.8, 'w2': -0.8},
        'bias': {'logic_and': -0.2, 'logic_nand': 0.3}
    }
    dataset = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    and_gate = compute('logic_and', logic, dataset)
    nand_gate = compute('logic_nand', logic, dataset)
    print("\nTruth Table for AND Gate")
    print("X1\tX2\tY")
    for data, out in zip(dataset, and_gate[1]):
         print(f"{data[1]}\t{data[2]}\t{out}")
    print("\nTruth Table for NAND Gate")
    print("X1\tX2\tY")
    for data, out in zip(dataset, nand_gate[1]):
        print(f"{data[1]}\t{data[2]}\t{out}")
  if __name__ == "__main__":
    main()

        
         
        
