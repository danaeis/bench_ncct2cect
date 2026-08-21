import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
# Upstream leaves batchSize/serial_batches/no_flip at their (train-oriented)
# defaults for test.py, unlike ResViT/CycleGAN's test.py which force them.
# util.tensor2im (called from get_current_visuals) only ever reads batch index
# 0, so any batchSize > 1 here would silently drop every other sample in the
# batch rather than error — forced here instead of trusted to the caller.
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
