# from unittest import mock
# import torch

# from src.www.app import synthesize

# @mock.patch('src.www.app.feature_model.infer', torch.FloatTensor(3, 4, 5))
# @mock.patch('src.www.app.signal_model.infer', (torch.FloatTensor(3), torch.FloatTensor(3), None))
# @mock.patch('src.www.app.text_encoder.decode')
# @mock.patch('src.www.app.text_encoder.encode')
# @mock.patch('src.www.app.flask.request.get_json')
# def test_synthesize(_, __, mock_decode, mock_encode, mock_get_json):
#     mock_get_json.return_value = {'text': 'This is a test.', 'isHighFidelity': True}
#     mock_encode.return_value = torch.LongTensor([1, 2, 3, 4, 5, 3, 4, 5, 6, 4, 7, 8, 4, 7, 9])
#     mock_decode.return_value = 'This is a test.'
