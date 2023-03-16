import torch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a=torch.randn((4,3,2,2))
    print(a)
    print(torch.mean(a,dim=0))
    print(torch.mean(a, dim=0).size())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
