import imageio
import os

def get_ratio(mel):
    """ Returns ratio of mel with low amplitude (essentially silent)

    :param mel: np.array of mel in image scale ([0,1])
    :return: Ratio between low amplitude pixels and overall size
    """
    low = (mel < 0.25).sum()
    total = mel.size
    return low/total

def save_image(img, dest_path, file_name):
    """ Saves an image

    :param img: np.array of uint8 image data
    :param dest_path: Path to save to
    :param file_name: File name
    """
    path = os.path.join(dest_path, f"{file_name}.png")
    imageio.imwrite(path, img)

def parse_file_id(file_id):
    """ Get rid of [] surrounding file_id

    :param file_id: File ID string formatted as in the csv (e.g. '[01371]'
    :return: file_id without brackets
    """
    file_id = str(file_id)
    file_id = file_id.replace("[", "")
    file_id = file_id.replace("]", "")
    return file_id