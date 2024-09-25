from m2t.data_modules import combine_emb_instruct_data
from functools import partial
from m2t.data_modules import webdataset_element_to_conversation
from m2t.data_modules import preprocess_multimodal_mappable


def Test_combine_emb_instruct_data():

    urls = ['datasets/training/1788-start0']
    src = combine_emb_instruct_data(urls)

    # print('Keys of combined data')
    # print(src[0].keys())
    # print('\n')

    multimodal_cfg={}
    multimodal_cfg['sep_audio_conv_front']=False
    multimodal_cfg['use_audio_start_end']=True

    elem = webdataset_element_to_conversation(src)

    _preprocess_multimodal = partial(preprocess_multimodal_mappable, multimodal_cfg=multimodal_cfg)

    mm1 = map(_preprocess_multimodal, elem)
    for a in mm1:
        print(a)

# Test_combine_emb_instruct_data()

def Test_read_webdataset_local():

    import webdataset as wds
    from m2t.data_modules import process_pkl_sample

    urls = ['datasets/training/tars/temp_train1_archive.tar']

    dataset = wds.WebDataset(urls)
    dataset = dataset.map(process_pkl_sample)    

    from m2t.data_modules import webdataset_element_to_conversation

    dataset = dataset.compose(webdataset_element_to_conversation)

    print('Checking after compose...')
    for sample in dataset:
        print(sample)    

    multimodal_cfg={}
    multimodal_cfg['sep_audio_conv_front']=False
    multimodal_cfg['use_audio_start_end']=True
    from functools import partial
    from m2t.data_modules import preprocess_multimodal_mappable
    _preprocess_multimodal = partial(preprocess_multimodal_mappable, multimodal_cfg=multimodal_cfg)

    a = dataset.map(_preprocess_multimodal)

    print("checking after map multimodal...")
    for sample in a:
        print(sample)

Test_read_webdataset_local()
