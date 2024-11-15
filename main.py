from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import uuid
from collections import Counter
from math import log, floor
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import io
from flask import Response

app = Flask(__name__)
app.secret_key = 'PUT_YOUR_KEY_HERE'
app.config["SESSION_PERMANENT"] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 7200
app.config["SESSION_TYPE"] = "filesystem"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
Session(app)
def process_input(user_input,typeof):
    if typeof=="Entropy":
        og=user_input.lower()
        data=list(dict(Counter(og)).values())
        p=[el/len(og) for el in data]
        entropy_values=[pi*log(pi,2) for pi in p]
        return (user_input,str(-sum(entropy_values)))
    elif typeof=="Entropy_of_pair":
        og=user_input.lower()
        pairs=[og[i]+og[i+1] for i in range(len(og)-1)]
        data=list(dict(Counter(pairs)).values())
        p=[el/len(pairs) for el in data]
        entropy_values=[pi*log(pi,2) for pi in p]
        return (user_input,str(-sum(entropy_values)))
    elif typeof=="RLE":
        og=user_input.upper()
        i=0
        temp_i=0
        inarow=0
        new_str=[]
        while i<len(og):
            new_str+=og[i]
            if og[i]=='#':
                inarow+=1
            else:
                if inarow>=1:
                    new_str[temp_i-1]=f'#{inarow}{og[i-1]}'
                    del new_str[temp_i-inarow:temp_i-1]
                    temp_i-=inarow-1
                inarow=0
            i+=1
            temp_i+=1
            if i==len(og):
                if inarow>=1:
                    new_str[temp_i-1]=f'#{inarow}{og[i-1]}'
                    del new_str[temp_i-inarow:temp_i-1]
                    temp_i-=inarow-1
                inarow=0
        og=''.join(new_str)
        i=0
        temp_i=0
        inarow=1
        new_str=[]
        while i<len(og):
            new_str+=og[i]
            if i==0:
                i+=1
                temp_i+=1
                continue
            if og[i-1]==og[i]:
                inarow+=1
            else:
                if inarow>=4:
                    new_str[temp_i-1]=f'#{inarow}{og[i-1]}'
                    del new_str[temp_i-inarow:temp_i-1]
                    temp_i-=inarow-1
                inarow=1
            i+=1
            temp_i+=1
            if i==len(og):
                if inarow>=4:
                    new_str[temp_i-1]=f'#{inarow}{og[i-1]}'
                    del new_str[temp_i-inarow:temp_i-1]
                    temp_i-=inarow-1
                inarow=1
        ESCAPE=''.join(new_str)
        og=user_input.upper()
        encoded = []
        i = 0
        while i < len(og):
            count = 1
            while i + 1 < len(og) and og[i] == og[i + 1]:
                count += 1
                i += 1
            if count >= 2:
                encoded.append(f"{count}{og[i]}")
            else:
                start = i
                count = 0
                while i < len(og) and (i + 1 == len(og) or og[i] != og[i + 1]):
                    count += 1
                    i += 1
                    if count == 127:
                        break
                encoded.append(f"{128 + count}{og[start:i]}")
                i -= 1
            i += 1
        JUMP=''.join(map(str, encoded))
        return (user_input,ESCAPE,JUMP)
    elif typeof=='LZW':
        uncompressed=user_input
        dict_size=256
        dictionary={chr(i): i for i in range(dict_size)}
        w=""
        compressed=[]
        for c in uncompressed:
            wc=w+c
            if wc in dictionary:
                w=wc
            else:
                compressed+=[dictionary[w]]
                dictionary[wc]=dict_size
                dict_size+=1
                w=c
        if w:
            compressed+=[dictionary[w]]
        return (user_input,'|'.join(map(str,compressed)))
    elif typeof=="Huffman(BACKUP)":
        class Tree(object):
            def __init__(self, l=None, r=None):
                self.l=l
                self.r=r
            def child(self):
                return (self.l, self.r)
            def nodes(self):
                return (self.l, self.r)
            def __str(self):
                return f"{self.l}_{self.r}"
        def huffman(node,l,string):
            if type(node) is str:
                return {node: string}
            (l, r) = node.child()
            d={}
            d.update(huffman(l,True,string+'0'))
            d.update(huffman(r,False,string+'1'))
            return d
        string=user_input.upper()
        freq={}
        for c in string:
            if c in freq:
                freq[c]+=1
            else:
                freq[c]=1
        freq = sorted(freq.items(), key=lambda x: x[1],reverse=True)
        nodes=freq
        while len(nodes) > 1:
            (key1, c1) = nodes[-1]
            (key2, c2) = nodes[-2]
            nodes = nodes[:-2]
            node = Tree(key1, key2)
            nodes.append((node, c1+c2))
            nodes=sorted(nodes,key=lambda x: x[1], reverse=True)
        huffman_code=huffman(nodes[0][0],True,'')
        new_string=string
        for key,value in huffman_code.items():
            new_string=new_string.replace(key,str(value)+'|')
        return (user_input,str(huffman_code),new_string[:-1])
    elif typeof=="Huffman":
        fig=plt.figure()
        class Tree(object):
            def __init__(self, l=None, r=None):
                self.l = l
                self.r = r
            def child(self):
                return (self.l, self.r)
            def __str__(self):
                return f"{self.l}_{self.r}"
        def huffman(node,l,string):
            if type(node) is str:
                return {node: string}
            (l, r) = node.child()
            d={}
            d.update(huffman(l,True,string+'0'))
            d.update(huffman(r,False,string+'1'))
            return d
        def plot_huffman_tree(node, pos=None, x=0, y=0, layer=1, G=None):
            if G is None:
                G = nx.DiGraph()
            if pos is None:
                pos = {}
            node_label = str(node) if isinstance(node, Tree) else node
            if type(node) is str:
                pos[node] = (x, y)
                G.add_node(node)
            else:
                pos[node_label] = (x, y)
                G.add_node(node_label)

                l, r = node.child()
                G.add_edge(node_label, str(l) if isinstance(l, Tree) else l)
                G.add_edge(node_label, str(r) if isinstance(r, Tree) else r)

                plot_huffman_tree(l, pos=pos, x=x - 1 / layer, y=y - 1, layer=layer + 1, G=G)
                plot_huffman_tree(r, pos=pos, x=x + 1 / layer, y=y - 1, layer=layer + 1, G=G)
            return G, pos
        string=user_input.upper()
        freq = {}
        for c in string:
            if c in freq:
                freq[c] += 1
            else:
                freq[c] = 1

        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        nodes = freq

        while len(nodes) > 1:
            (key1, c1) = nodes[-1]
            (key2, c2) = nodes[-2]
            nodes = nodes[:-2]
            node = Tree(key1, key2)
            nodes.append((node, c1 + c2))
            nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

        huffman_code = huffman(nodes[0][0], True, '')
        new_string=string
        for key,value in huffman_code.items():
            new_string=new_string.replace(key,str(value)+'|')
        G, pos = plot_huffman_tree(nodes[0][0])
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        session['huffman_tree_bytes']=output.getvalue()
        if len(user_input)==1 or len(Counter(string).items())==1:
            return (user_input,{list(Counter(string).items())[0][0]:0},string.replace(list(Counter(string).items())[0][0],'0|')[:-1])
        else:
            return (user_input,str(huffman_code),new_string[:-1])
    elif typeof=="Arithmetic_coding":
        txt=user_input
        new_dict={}
        global_intervals={}
        start=0
        for key,value in Counter(txt).items():
            new_dict[key]=value/len(txt)
            global_intervals[key]=[start,start+new_dict[key]]
            start=start+new_dict[key]
        new_interval=[0,1]
        for letter in txt:
            difference=new_interval[1]-new_interval[0]
            isitfirst=True
            for key,value in new_dict.items():
                if isitfirst:
                    new_interval=[new_interval[0],new_interval[0]+(value*(difference))]
                    isitfirst=False
                else:
                    new_interval=[new_interval[1],new_interval[1]+(value*(difference))]
                if key==letter:
                    break
        return (user_input,new_interval, (new_interval[0]+new_interval[1])/2)
    elif typeof=="Shannon_Fano":
        def shannon_fano_recursive(symbols, codes):
            if len(symbols) == 1:
                symbol = symbols[0][0]
                return {symbol: codes}
            total_weight = sum(weight for symbol, weight in symbols)
            cumulative_weight = 0
            split_index = 0
            for i, (symbol, weight) in enumerate(symbols):
                cumulative_weight += weight
                if cumulative_weight >= floor(total_weight / 2):
                    split_index = i
                    break
            left_symbols = symbols[:split_index+1]
            right_symbols = symbols[split_index+1:]
            left_codes = shannon_fano_recursive(left_symbols, codes + '0')
            right_codes = shannon_fano_recursive(right_symbols, codes + '1')
            return {**left_codes, **right_codes}
        def shannon_fano_encoding(string):
            frequency = Counter(string)
            sorted_symbols = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
            codes = shannon_fano_recursive(sorted_symbols, '')
            return codes
        txt=user_input.upper()
        if len(txt)==1 or len(Counter(txt).items())==1:
            return (user_input,{list(Counter(txt).items())[0][0]:0},txt.replace(list(Counter(txt).items())[0][0],'0|')[:-1])
        else:
            res=shannon_fano_encoding(txt)
            new_string=txt
            for key,value in res.items():
                new_string=new_string.replace(key,str(value)+'|')
            return (user_input,str(res),new_string[:-1])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/entropy')
def entropy():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_entropy0', 'Nothing yet.')
    result2 = session.get('result_entropy1', 'Nothing yet.')
    return render_template('entropy.html', result_entropy0=result1, result_entropy1=result2)

@app.route('/entropy_of_pair')
def entropy_of_pair():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_entropy_of_pair0', 'Nothing yet.')
    result2 = session.get('result_entropy_of_pair1', 'Nothing yet.')
    return render_template('entropy_of_pair.html', result_entropy_of_pair0=result1,result_entropy_of_pair1=result2)

@app.route('/rle')
def rle():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_rle0', 'Nothing yet.')
    result2 = session.get('result_rle1', 'Nothing yet.')
    result3 = session.get('result_rle2', 'Nothing yet.')
    return render_template('rle.html', result_rle0=result1, result_rle1=result2, result_rle2=result3)

@app.route('/lzw')
def lzw():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_lzw0', 'Nothing yet.')
    result2 = session.get('result_lzw1', 'Nothing yet.')
    return render_template('lzw.html', result_lzw0=result1,result_lzw1=result2)

@app.route('/tree.png')
def plot_huffman_tree():
    fig=session.get('huffman_tree_bytes',None)
    if fig:
        return Response(fig,mimetype='image/png')
    else:
        output = io.BytesIO()
        Image.open('static/huffman_empty_1x1.png').save(output, format='PNG')
        return Response(output.getvalue(),mimetype='image/png')

@app.route('/huffman')
def huffman():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_huffman0', 'Nothing yet.')
    result2 = session.get('result_huffman1', 'Nothing yet.')
    result3 = session.get('result_huffman2', 'Nothing yet.')
    return render_template('huffman.html', result_huffman0=result1, result_huffman1=result2, result_huffman2=result3)

@app.route('/arithmetic_coding')
def arithmetic_coding():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_arithmetic_coding0', 'Nothing yet.')
    result2 = session.get('result_arithmetic_coding1', 'Nothing yet.')
    result3 = session.get('result_arithmetic_coding2', 'Nothing yet.')
    return render_template('arithmetic_coding.html', result_arithmetic_coding0=result1, result_arithmetic_coding1=result2, result_arithmetic_coding2=result3)

@app.route('/shannon_fano')
def shannon_fano():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_shannon_fano0', 'Nothing yet.')
    result2 = session.get('result_shannon_fano1', 'Nothing yet.')
    result3 = session.get('result_shannon_fano2', 'Nothing yet.')
    return render_template('shannon_fano.html', result_shannon_fano0=result1, result_shannon_fano1=result2, result_shannon_fano2=result3)

@app.route('/color_histogram')
def color_histogram():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    result1 = session.get('result_entropy0', 'Nothing yet.')
    result2 = session.get('result_entropy1', 'Nothing yet.')
    '''session['color_histogram_image']=session.get('color_histogram_image',None)
    session['color_histogram_bins']=session.get('color_histogram_bins',40)'''
    session['image_bytes']=session.get('image_bytes',None)
    return render_template('color_histogram.html', result_entropy0=result1, result_entropy1=result2)

@app.route('/plot.png')
def plot_color_histogram():
    fig = session.get('image_bytes',None)
    if fig:
        if fig!='Error!':
            return Response(fig, mimetype='image/png')
        else:
            output = io.BytesIO()
            Image.open('static/color_histogram_error.png').save(output, format='PNG')
            return Response(output.getvalue(),mimetype='image/png')
    else:
        output = io.BytesIO()
        Image.open('static/color_histogram_empty.png').save(output, format='PNG')
        return Response(output.getvalue(),mimetype='image/png')

def create_figure(image, bins):
    if image and bins and 1<=bins<=256:
        image=np.array(Image.open(image))
        histogram = np.zeros((bins, bins, bins))
        bin_size_without_rem = 256//bins
        rem = 256%bins
        if rem!=0:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    r, g, b = image[i, j][:3]
                    if r<rem*(bin_size_without_rem+1):
                        r_bin=r//(bin_size_without_rem+1)
                    else:
                        r_bin=(r-rem*(bin_size_without_rem+1))//bin_size_without_rem + rem
                    if g<rem*(bin_size_without_rem+1):
                        g_bin=g//(bin_size_without_rem+1)
                    else:
                        g_bin=(g-rem*(bin_size_without_rem+1))//bin_size_without_rem + rem
                    if b<rem*(bin_size_without_rem+1):
                        b_bin=b//(bin_size_without_rem+1)
                    else:
                        b_bin=(b-rem*(bin_size_without_rem+1))//bin_size_without_rem + rem
                    histogram[r_bin, g_bin, b_bin] += 1
        else:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    r, g, b = image[i, j][:3]
                    r_bin = r // bin_size_without_rem
                    g_bin = g // bin_size_without_rem
                    b_bin = b // bin_size_without_rem
                    histogram[r_bin, g_bin, b_bin] += 1
        histogram /= image.shape[0] * image.shape[1]
        r_hist = histogram.sum(axis=(1, 2))
        g_hist = histogram.sum(axis=(0, 2))
        b_hist = histogram.sum(axis=(0, 1))

        bin_edges=np.arange(0,256,256/bins)
        fig=plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        if bins<100:
            plt.bar(bin_edges, r_hist, width=(256 / bins), color='red', edgecolor='black')
        else:
            plt.bar(bin_edges, r_hist, width=(256 / bins), color='red')
        plt.title('Red Channel')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.subplot(1, 3, 2)
        if bins<100:
            plt.bar(bin_edges, g_hist, width=(256 / bins), color='green', edgecolor='black')
        else:
            plt.bar(bin_edges, g_hist, width=(256 / bins), color='green')
        plt.title('Green Channel')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.subplot(1, 3, 3)
        if bins<100:
            plt.bar(bin_edges, b_hist, width=(256 / bins), color='blue', edgecolor='black')
        else:
            plt.bar(bin_edges, b_hist, width=(256 / bins), color='blue')
        plt.title('Blue Channel')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.tight_layout()
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return output.getvalue()
    else:
        return "Error!"

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form.get('input-box')
    if user_input:
        if len(user_input)<=512:
            if request.form.get('typeof')=='Color_histogram':
                image=request.files['input-file']
                if image:
                    try:
                        session['image_bytes']=create_figure(image, int(user_input))
                    except ValueError:
                        session['image_bytes']=None
                    '''try:
                        session['color_histogram_image']=np.array(Image.open(image)).tolist()
                        session['color_histogram_bins']=int(user_input)
                    except ValueError:
                        session['color_histogram_image']=None
                        session['color_histogram_bins']=40'''

            else:
                result = process_input(user_input,request.form.get('typeof'))
                if type(result)!=tuple:
                    session[f'result_{request.form.get("typeof").lower()}'] = result
                else:
                    for i in range(len(result)):
                        session[f'result_{request.form.get("typeof").lower()}{i}'] = result[i]
    
    return redirect(url_for(request.form.get('typeof').lower()))

if __name__ == '__main__':
    app.run()
