# Basic module

# Torch and visulization
from torchvision      import transforms

# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

import time

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet')

    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet), False(model==ACM)')

    # data and pre-process
    parser.add_argument('--img_dir', type=str,
                        help='img_dir')
    # parser.add_argument('--img_demo_index', type=str,default='target3',
    #                     help='target1, target2, target3')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')

    args = parser.parse_args()
    # the parser
    return args

class Inference(object):
    def __init__(self, args):
        # Initial
        self.args = args
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Preprocess and load data
        self.transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        # Choose and load model (this paper is finished by one GPU)
        model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        model           = model.cuda()
        model.apply(weights_init_xavier)
        self.model      = model

        # Load Checkpoint
        checkpoint      = torch.load('pretrain_DNANet_model.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()

    def _demo_sync_transform(self, img):
        base_size = self.args.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self, img):
        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img

    def predict(self, file_path):
        #Open Image
        img = Image.open(file_path).convert('RGB')
        
        #Preprocess
        img = self.img_preprocess(img)

        #Prediction
        img = img.cuda()
        img = torch.unsqueeze(img,0)

        preds = self.model(img)
        pred  = preds[-1]

        #Visualization
        save_Pred_GT_visulize(pred, file_path, args.suffix, args.base_size, visualize=False)



def main(args):
    inference = Inference(args)

    # Specify the directory containing the images
    directory = args.img_dir

    # Loop over each file in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith((inference.args.suffix)):  # Add or remove file types as needed
            file_path = os.path.join(directory, filename)
            start_time = time.time()
            inference.predict(file_path)

            time_mark = round((time.time() - start_time), 2)
            fps = round((1 / time_mark), 2)

            print("Image: " + file_path, "Time: ", time_mark, 's', "FPS: ", fps)

if __name__ == "__main__":
    args = parse_args()
    main(args)





