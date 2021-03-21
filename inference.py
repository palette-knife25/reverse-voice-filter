import os
import glob
import torch
import librosa
import argparse

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def main(args, hp):
    with torch.no_grad():
        model = VoiceFilter(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)
        dvec_wav, _ = librosa.load(args.reference_file, sr=16000)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        mixed_wav, _ = librosa.load(args.mixed_file, sr=16000)
        mag, phase = audio.wav2spec(mixed_wav)
        mag = torch.from_numpy(mag).float().cuda()

        mag = mag.unsqueeze(0)
        mask = model(mag, dvec)

        if args.mask == "org":
            est_mag = mag * mask
        elif args.mask == "inv_div":
            est_mag = args.factor* mag/mask
        elif args.mask == "inv_sub":
            est_mag = mag * (torch.ones_like(mask) - mask)/args.factor

        est_mag = est_mag[0].cpu().detach().numpy()
        est_wav = audio.spec2wav(est_mag, phase)

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'result.wav')
        librosa.output.write_wav(out_path, est_wav, sr=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_file', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')
    parser.add_argument('--mask', type=str, required=True,
                        help='how to apply mask: "org" - spec*mask; "inv_div" - factor*spec/mask; "inv_sub" - spec*(1-mask)/factor')
    parser.add_argument('--factor', type=float, required=True, default=None,
                        help="factor for mask application; if --mask org - None, else - preferably 0.7")

    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
