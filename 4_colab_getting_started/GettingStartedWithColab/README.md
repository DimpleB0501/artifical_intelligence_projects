# Brief Overview

This section contains my notes on getting started with google colab and two projects implemented in colaboratory.

* **edgeDetection.ipynb** - I have implemented a canny edge detector that processes image stored in your google drive and writes back the result (output image) from the edge detector to your drive. This project aims to familiarize with permitting Colab to access your google drive and on installing/ implementing numpy and opencv functions.

* **MNISTwithPytorchAndColab.ipynb** - This project is an implementation of LENET architecture to train/ test MNIST dataset using pytorch library. The code is structured to access GPU mode in Colab during both training and testing. You can switch on the GPU mode by clicking in Colab Project on 
**Runtime** &rarr; **Change Runtime type** &rarr; **Hardware accelerator** &rarr; Select **None/GPU/TPU**

     

# Getting Started with Google Colab
### Installing colaboratory 
* Go to your google drive Click on My Drive &rarr; More &rarr; Connect more apps
![Install colab](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/1_AddingCollab.png) 
* Search for Colab in the search bar and CONNECT.
![Add colab](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/12_ColabSearch.png) 
* Now you can find Colaboratory in your apps.
![Find colab in drive](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/2_colaboratoryInDrive.png) 
* And like normal drive applications by clicking on **NEW** &rarr; **More** &rarr; **Colaboratory** you can start a new project or open an existing project by 
![Start a colab project](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/13_ExistingProject.png)

### Colab Basics
* Goto Tools &rarr; Settings &rarr; Site  &rarr; Site.
In Site you can select your theme and it is really important to adjust the indentation width in spaces to conform to python3 norms.
![Preferences](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/3_Preferences.png)

* Switch to GPU mode 
Click in your Colab Project on 
**Runtime** &rarr; **Change Runtime type** &rarr; **Hardware accelerator** &rarr; Select **None/GPU/TPU**
![Accelerator select](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/4_SwitchingGPUmode.png)
Note: The GPU resets after every 12 Hours in collab. Therefore it is important to keep on saving checkpoints to resume training later.

* Code or text cell.
Like jupyter notebook each cell can be used as a code or text cell. 
   -  You can __insert__ cells by adding them via clicking  on the + sign here
     ![Insert cell 1](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/5_AddingCells.png)
     OR
     
      By clicking on Insert     
      ![Insert cell 2](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/6_Bars.png)
  - To __delete__ a cell
   ![Delete cell](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/7_DeleteCell.png)
  - To __run__ a cell
   ![Run cell 1](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/8_RunningACell.png)
   OR
   Goto Runtime and select **Run all** or from other desired options.
   ![Run cell 2](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/9_FromRuntime.png)
  - Text cell as a wrapper/ header
    Just like jupyter notebooks you can have headers and you can add documentation or codes under the header. 
    To declare a header in google colab Select a text cell and add __#__ sign prior to your heading.
    ![Add header](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/14_addheader.png)
    this will cause a section
    ![Header 1](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/15_header1.png)
    which is expanded as
    ![Header 2](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/16_header2.png)
    
*  Running bash commands or Installing libraries in Colab.
These can be done in the code cell by starting the command with !

     ``!git clone [clone url]`` clones a git respository

     ``!wget [url] -p drive/[Folder Name]`` downloads from web

     ``!pip install [package name]`` or ``!apt-get install [package name]`` installs libraries.

     ``!ls `` or ``!mkdir`` excecutes shell commands
![Commands](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/10_RunningCommands.png)
* Prior to running you code make sure you are connected to the host by checking the following
![ConnectToHost](https://github.com/DimpleB0501/SecureNPrivateAIScholarshipCodes/blob/master/GettingStartedWithColab/ReadmeImages/11_connectedToHost.png)

* You can save your colab project in github by selecting **File** &rarr; **Save a copy in GitHub**

### References
* [Colab begineers guide](https://medium.com/lean-in-women-in-tech-india/google-colab-the-beginners-guide-5ad3b417dfa)
* [GPU in colab](https://www.kdnuggets.com/2018/02/google-colab-free-gpu-tutorial-tensorflow-keras-pytorch.html/2)
* [Pytorch introduction with Colab](https://colab.research.google.com/drive/1gJAAN3UI9005ecVmxPun5ZLCGu4YBtLo)


