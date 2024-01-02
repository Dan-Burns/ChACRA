import numpy as np
#TODO add option to globally define chacra color scheme
# and import into the plot and pymol modules
chacra_colors = ['#ff0000ff','#00bfffff','#00b730ff','#7400ffff','#ff831dff','#ff00ffff','#fad300ff','#434343ff','#b2a8b1c0','#b3ad45c0']
cherenkov_blue = '#00bfff'
legacy_blue = '#02a8f8ff'
orange = '#ff831dff'

# adopted next two functions from https://medium.com/@BrendanArtley/matplotlib-color-gradients-21374910584b
def hex_to_RGB(hex_str,alpha=False):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    if alpha == True:
        if len(hex_str) == 7:
            # no transparency so append 0 transparency
            hex_str = hex_str + "ff"
            return [int(hex_str[i:i+2], 16) for i in range(1,8,2)]
        else:
            return [int(hex_str[i:i+2], 16) for i in range(1,8,2)] 
    else:
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n, return_rgb=False, alpha=True):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1,alpha=alpha))/255
    c2_rgb = np.array(hex_to_RGB(c2,alpha=alpha))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    if return_rgb:
        return [[int(round(val*255)) for val in item] for item in rgb_colors]
    else:
        # return hex colors
        return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def rgb_to_hex(rgb):
    return "#" + "".join([format(int(round(val)), "02x") for val in rgb])

# adopted from https://stackoverflow.com/questions/52992900/how-to-blend-two-rgb-colors-front-and-back-based-on-their-alpha-channels
def mix_color(colorRGBA1, colorRGBA2):
    alpha = 255 - ((255 - colorRGBA1[3]) * (255 - colorRGBA2[3]) / 255)
    red   = (colorRGBA1[0] * (255 - colorRGBA2[3]) + colorRGBA2[0] * colorRGBA2[3]) / 255
    green = (colorRGBA1[1] * (255 - colorRGBA2[3]) + colorRGBA2[1] * colorRGBA2[3]) / 255
    blue  = (colorRGBA1[2] * (255 - colorRGBA2[3]) + colorRGBA2[2] * colorRGBA2[3]) / 255
    return (int(red), int(green), int(blue), int(alpha))

# example
# r_grad = get_color_gradient('#ff000000','#ff0000ff',100, return_rgb=True,alpha=True)
# b_grad = get_color_gradient('#02a8f800','#02a8f8ff',100, return_rgb=True, alpha=True)
# rgb = get_color(r_grad[50],b_grad[50])
# rgb_to_hex(get_color(r_grad[50],b_grad[50]))


class ANSI():
    '''
    credit: https://www.geeksforgeeks.org/how-to-add-colour-to-text-python/

    example_ansi = ANSI.background(
    97) + ANSI.color_text(35) + ANSI.style_text(4) + " TESTE ANSI ESCAPE CODE"
    print(example_ansi)
    '''
    def background(code):
        return "\33[{code}m".format(code=code)
 
    def style_text(code):
        return "\33[{code}m".format(code=code)
 
    def color_text(code):
        return "\33[{code}m".format(code=code)
    
wheel_of_dharma_ascii = '\u2638'
 
red_ascii = 31
green_ascii = 32
yellow_ascii = 33
blue_ascii = 34
magenta_ascii = 35
cyan_ascii = 36
white_ascii = 37
black_ascii = 90

dharma_string = ANSI.color_text(35) + wheel_of_dharma_ascii