# Information Theory
![Image](https://github.com/user-attachments/assets/15c0703a-2b94-47e1-814f-74b1f9212382)

Information Theory studies the principles of transmission, processing and storage of information, as well as methods of optimal data coding to ensure their reliability and efficiency. This website explores many themes and topics that are involved with information theory and provides as a result, a fully functional website with main compression methods and other algorithms being implemented there, for people to use and test. It also includes necessary information for each of them to understand.

This project provides the implementation of these encoding algorithms:
- Entropy
- Entropy of pair
- RLE
- LZW
- Huffman algorithm
- Arithmetic coding
- Shannon-Fano algorithm

And as an additional algorithm denoted as "Project" on the website:
- Color Histogram Encoding

## Installation:

The application requires Python 3.10, other versions were not tested.

It was also programmed and tested on Native Windows 11, it is unknown if this will work with other operating systems.

Clone the repository from git clone: [https://github.com/ManulProgramming/Information_Theory_Encoding_Algorithms](https://github.com/ManulProgramming/Information_Theory_Encoding_Algorithms).

Go to the /Information_Theory_Encoding_Algorithms and install requirements using pip:

```bash
pip install -r requirements.txt
```

## Usage:

Before running the application itself, using either a development server or production WSGI server, you need to put your own secret key in the application, by changing the 15th line of code:
```python
app.secret_key = 'PUT_YOUR_KEY_HERE'
```

After that, you can run the server using a simple bash command:
```bash
python main.py
```

Or use the WSGI server of your choice to run it in the production server.

## Notes:

This application is created for educational purposes. It should not be considered as a serious tool, but rather as a basic Python project.


Author: Dmitriy Boyarkin IT2-2312

## License:

[MIT](https://github.com/ManulProgramming/Information_Theory_Encoding_Algorithms/blob/main/LICENSE)
