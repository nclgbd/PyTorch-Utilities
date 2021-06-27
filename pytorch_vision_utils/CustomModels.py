"""Contains all of the available models for customization. Wrapper for another module I've been messing with.
"""

from pretrainedmodels import alexnet
from pretrainedmodels import densenet121
from pretrainedmodels import densenet169
from pretrainedmodels import densenet201
from pretrainedmodels import densenet161
from pretrainedmodels import resnet18
from pretrainedmodels import resnet34
from pretrainedmodels import resnet50
from pretrainedmodels import resnet101
from pretrainedmodels import resnet152
from pretrainedmodels import inceptionv3
from pretrainedmodels import squeezenet1_0
from pretrainedmodels import squeezenet1_1
from pretrainedmodels import vgg11
from pretrainedmodels import vgg11_bn
from pretrainedmodels import vgg13
from pretrainedmodels import vgg13_bn
from pretrainedmodels import vgg16
from pretrainedmodels import vgg16_bn
from pretrainedmodels import vgg19_bn
from pretrainedmodels import vgg19

from pretrainedmodels.models.xception import Xception
from pretrainedmodels.models.mobilenetv2 import MobileNetV2


### Specially made homegrown abstraction of annoying torch shit


class CustomVGG19():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the vgg19 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = vgg19(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomVGG16():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the vgg16 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = vgg16(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomVGG13():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the vgg13 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = vgg13(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomVGG11():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the vgg11 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = vgg11(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomSqueezeNet1_1():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the squeezenet1_1 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = squeezenet1_1(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomSqueezeNet1_0():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the squeezenet1_0 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = squeezenet1_0(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomInceptionV3():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the inceptionv3 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = inceptionv3(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomResNet152():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the resnet152 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = resnet152(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomResNet101():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the resnet101 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = resnet101(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomResNet50():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the resnet50 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = resnet50(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomResNet34():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the resnet34 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = resnet34(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomResNet18():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the resnet18 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = resnet18(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomDenseNet161():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the densenet161 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = densenet161(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomDenseNet201():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the densenet201 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = densenet201(num_classes=num_of_classes)
        
        if debug:
            print(self.model)


class CustomDenseNet169():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the densenet169 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = densenet169(num_classes=num_of_classes)
        
        if debug:
            print(self.model)

         
class CustomDenseNet121():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the densenet121 function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = densenet121(num_classes=num_of_classes)
        
        if debug:
            print(self.model)
            
            
            
class CustomAlexNet():
    
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the alexnet function to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """  
        self.model = alexnet(num_classes=num_of_classes)
        
        if debug:
            print(self.model)
      
      
   
class CustomXception(Xception):
        
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the Xception class to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """        
        super(CustomXception, self).__init__()
        if debug:
            print(self)


class CustomMobileNetV2(MobileNetV2):
        
    def __init__(self, num_of_classes=2, debug=False):
        """
        Wrapper around the mobilenetv2 class to change the classifier from 1000 to 2 and adds a debug functionality.

        Attributes
        ----------
        `num_of_classes` : int, optional\n
            The number of classes being predicted on, by default 2.
        `debug` : bool, optional\n
            Boolean representing whether debug mode is on or off, by default False.
        """        
        super(MobileNetV2, self).__init__()
        if debug:
            print(self)
            
        