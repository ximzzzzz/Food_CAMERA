#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import os
import sys
import cv2
from PIL import Image
import easydict
sys.path.append('./Whatiswrong')
sys.path.append('./Nchar_clf')
sys.path.append('./EFIFSTR_torch')
sys.path.append('./Scatter')
import augs
import Model
import efifstr_utils
import Nchar_utils
import Extract
import utils
import evaluate
import torch.nn.functional as F
from torch.utils.data import *
import easydict
import torchvision
import pickle
import time
import os
from albumentations import GaussNoise, IAAAdditiveGaussianNoise, Compose, OneOf
from albumentations.pytorch import ToTensor
import albumentations
import re


# In[2]:


import importlib
importlib.reload(evaluate)


# In[3]:


opt = easydict.EasyDict({
    'experiment_name' : f'{utils.SaveDir_maker(base_model = "EFIFSTR_nchar", base_model_dir="./models")}',
    'manualSeed' : 1111,
    'saved_model' : 'EFIFSTR_nchar_1124/1/best_accuracy_86.13.pth',
#     'saved_model' : '',
    'num_workers' : 4,
    'TPS' : True,
    'num_fiducial' : 20,
    'grad_clip' : 5,
    'val_interval' : 3000,
    'num_epoch' : 10,
    'lr' : 1, 'rho' : 0.95, 'eps' : 1e-8,
    'max_length' : 23,
    'num_fonts' : 104,
    'img_h' : 48, 'img_w' : int((48/4)*23), # for TPS
    'fmap_dim' : 512, 'enc_dim' : 512, 'attn_dim' : 512, 'dec_dim' : 512,
    'batch_size' : 64 ,
    'character' : '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읩읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝'
                        } )
device = torch.device('cuda') 
opt.num_classes = len(opt.character)
opt.img_w_max = (opt.img_h/4) * opt.max_length # for generator
# opt.img_w_max = 160
converter = efifstr_utils.LabelConverter(opt.character, device)


# In[4]:


with open('./dataset_syllable_180', 'rb') as file:
    data = pickle.load(file)
# with open('./dataset_light', 'wb') as file:
#     pickle.dump(data[:1000], file)


# In[5]:


new_data = []
for path, label in data:
    if re.compile('자와틍아주').match(label):
        continue
    else:
        path = path.replace('/home','')
    new_data.append([path, label])

train_data = new_data[ : int(len(data) * 0.998)]
test_data = new_data[ int(len(data) * 0.998): ]

# In[5]:


nchar_dataset = Nchar_utils.CustomDataset_syllable(train_data, device=device, resize_shape=(opt.img_h, opt.img_w), max_length = opt.max_length, is_module=True, is_train=True)
data_loader = DataLoader(nchar_dataset, batch_size = opt.batch_size , pin_memory=True, drop_last=True, num_workers = opt.num_workers)
valid_dataset = Nchar_utils.CustomDataset_syllable(test_data, device=device, resize_shape=(opt.img_h, opt.img_w), is_module=True, is_train=False)
valid_loader = DataLoader(valid_dataset, batch_size = opt.batch_size , pin_memory=True, drop_last=True, num_workers = opt.num_workers)


# In[6]:


# transformers = Compose([
#                         OneOf([
#                                   augs.VinylShining(1),
#                             augs.GridMask(num_grid=(15,15)),
#                             augs.RandomAugMix(severity=1, width=1)], p =0.7),
#                             ToTensor()
#                        ])
# train_custom = utils.Dataset_streamer(new_data[ : int(len(data) * 0.998)], resize_shape = (opt.img_h, opt.img_w), transformer=transformers)
# valid_custom = utils.Dataset_streamer(new_data[ int(len(data) * 0.998): ], resize_shape = (opt.img_h, opt.img_w), transformer=ToTensor())

# data_loader = DataLoader(train_custom, batch_size = opt.batch_size,  num_workers =opt.num_workers, shuffle=True, drop_last=True, pin_memory=True)
# valid_loader = DataLoader(valid_custom, batch_size = opt.batch_size,  num_workers=opt.num_workers, shuffle=True,  drop_last=True, pin_memory=True )


# In[7]:


def train(opt):
    model = Model.Basemodel(opt, device)
    
    # weight initialization
    for name, param, in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initializaed')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
                
        except Exception as e :
            if 'weight' in name:
                param.data.fill_(1)
            continue
            
    # load pretrained model
    if opt.saved_model != '':
        base_path = './models'
        print(f'looking for pretrained model from {os.path.join(base_path, opt.saved_model)}')
        
        try :
            model.load_state_dict(torch.load(os.path.join(base_path, opt.saved_model)))
            print('loading complete ')    
        except Exception as e:
            print(e)
            print('coud not find model')
            
    #data parallel for multi GPU
    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
    model.train() 
    
    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p : p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Tranable params : ', sum(params_num))

    loss_avg = utils.Averager()
    loss_avg_glyph = utils.Averager()
    
    # optimizer
    
    optimizer = optim.Adadelta(filtered_parameters, lr= opt.lr, rho = opt.rho, eps = opt.eps)
#     optimizer = torch.optim.Adam(filtered_parameters, lr=0.0001)
#     optimizer = SWA(base_opt)
#     optimizer = torch.optim.AdamW(filtered_parameters)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience = 2, factor= 0.5 )
#     optimizer = adabound.AdaBound(filtered_parameters, lr=1e-3, final_lr=0.1)

    # opt log
    with open(f'./models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '---------------------Options-----------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log +=f'{str(k)} : {str(v)}\n'
        opt_log +='---------------------------------------------\n'
        opt_file.write(opt_log)
        
    #start training
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    swa_count = 0
    
    for n_epoch, epoch in enumerate(range(opt.num_epoch)):
        for n_iter, data_point in enumerate(data_loader):
            
            image, labels = data_point 
            image = image.to(device)
            try:
                target, length = converter.encode(labels, batch_max_length = opt.max_length)
                batch_size = image.size(0)
            except Exception as e:
                print(f'{e}')
                continue

            logits, glyphs, embedding_ids = model(image, (target, length), is_train = True)
            
            recognition_loss = model.module.decoder.recognition_loss(logits.view(-1, opt.num_classes+2), target.view(-1))
            generation_loss = model.module.generator.glyph_loss(glyphs, target, length, embedding_ids, opt)
            
            cost = recognition_loss + generation_loss

            loss_avg.add(recognition_loss)
            loss_avg_glyph.add(generation_loss)
            
            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) #gradient clipping with 5
            optimizer.step()
            
            #validation
            if (n_iter % opt.val_interval == 0) & (n_iter!=0)   :
#                 & (n_iter!=0)
                elapsed_time = time.time() - start_time
                with open(f'./models/{opt.experiment_name}/log_train.txt', 'a') as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss_recog, valid_loss_glyph, current_accuracy, current_norm_ED, preds, confidence_score, labels,                         infer_time, length_of_data = evaluate.validation_efifstr(model, valid_loader, converter, opt)
                    model.train()

                    present_time = time.localtime()
                    loss_log = f'[epoch : {n_epoch}/{opt.num_epoch}] [iter : {n_iter*opt.batch_size} / {int(len(data) * 0.998)}]\n'+                    f'Train recognition loss : {loss_avg.val():0.5f}, Glyph loss : {loss_avg_glyph.val():0.5f}\nValid recogntion loss : {valid_loss_recog:0.5f}, Glyph loss : {valid_loss_glyph:0.5f}, Elapsed time : {elapsed_time:0.5f}, Present time : {present_time[1]}/{present_time[2]}, {present_time[3]+9} : {present_time[4]}'
                    loss_avg.reset()
                    loss_avg_glyph.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"current_norm_ED":17s}: {current_norm_ED:0.2f}'

                    #keep the best
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/best_accuracy_{round(current_accuracy,2)}.pth')

                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/best_norm_ED.pth')

                    best_model_log = f'{"Best accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'
                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log+'\n')

                    dashed_line = '-'*80
                    head = f'{"Ground Truth":25s} | {"Prediction" :25s}| Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

                    random_idx  = np.random.choice(range(len(labels)), size= 5, replace=False)
                    for gt, pred, confidence in zip(list(np.asarray(labels)[random_idx]), list(np.asarray(preds)[random_idx]), list(np.asarray(confidence_score)[random_idx])):
                        gt = gt[: gt.find('[s]')]
                        pred = pred[: pred.find('[s]')]

                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log+'\n')

#                 # Stochastic weight averaging
#                 optimizer.update_swa()
#                 swa_count+=1
#                 if swa_count % 3 ==0:
#                     optimizer.swap_swa_sgd()
#                     torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/swa_{swa_count}.pth')

        if (n_epoch) % 5 ==0:
            torch.save(model.module.state_dict(), f'./models/{opt.experiment_name}/{n_epoch}.pth')


# In[ ]:


os.makedirs(f'./models/{opt.experiment_name}', exist_ok=True)

# set seed
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)

# set GPU
cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

# if opt.num_gpu > 1:
#     print('-------- Use multi GPU setting --------')
#     opt.workers = opt.workers * opt.num_gpu
#     opt.batch_size = opt.batch_size * opt.num_gpu

train(opt)



