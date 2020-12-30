import torch 
import os 

from models_inference import LeNet_standard, LeNet_dropout, LeNet_manualdropout

def main():
    # create directory for storing models
    os.makedirs('onnx_mdls/', exist_ok=True)

    # create input
    dummy_input = torch.zeros(280 * 280 * 4)
    
    # convert LeNet_standard
    pytorch_model = LeNet_standard()
    ckpt_standard = torch.load('checkpoint/LeNet_stadard60.pth.tar')
    pytorch_model.load_state_dict(ckpt_standard['state_dict'])
    pytorch_model.eval()
    torch.onnx.export(pytorch_model, dummy_input, 'onnx_lenet_standard.onnx', verbose=True)
    
    # convert LeNet_dropout
    # pytorch_model = LeNet_dropout()
    # ckpt_standard = torch.load('checkpoint/LeNet_dropout60.pth.tar')
    # pytorch_model.load_state_dict(ckpt_standard['state_dict'])
    # pytorch_model.train()
    # torch.onnx.export(pytorch_model, dummy_input, 'onnx_mdls/onnx_lenet_dropout.onnx', verbose=True)

    # convert LeNet_manualdropout
    pytorch_model = LeNet_manualdropout()
    ckpt_standard = torch.load('checkpoint/LeNet_dropout60.pth.tar')
    pytorch_model.load_state_dict(ckpt_standard['state_dict'])
    pytorch_model.eval()
    torch.onnx.export(pytorch_model, dummy_input, 'onnx_mdls/onnx_lenet_manualdropout.onnx', verbose=True)


if __name__ == '__main__':
    main()
