#Created by Akmal

import sys
import telegram
import urllib.request as urllib2
import cv2
from PIL import Image
import torch
from torchvision import transforms
from vgg import VGG
from datasets import FER2013
from utils import eval, detail_eval
from face_detect.haarcascade import haarcascade_detect
import numpy as np
import argparse
import random
import emote
import time

from time import sleep


fromBotDir = 'images_from_bot'
toBotDir = 'images_to_bot'

#botName = 'suprisingly_bot'
#botToken = 'Your Bot Token'

botName = 'test4pyt_bot'
botToken = '1179400232:AAEXTovqtnKJsYWdEO77nVQ9qLGVMt1bHAE'	#Create bot with 'BotFather' from Telegram
baseUrl = 'https://api.telegram.org/bot%s' % botToken
longPoolingTimeoutSec = 60

sleepIntervalSec = 2
lastConsumedUpdateInit = 0	

##start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--trained_model", default = "./model/model_state.pth.tar", type= str,
                help = "Trained state_dict file path to open")
ap.add_argument("--model_name", default= "VGG19",type= str, help = "name model")
args = ap.parse_args()

classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
crop_size= 44

#Load model
trained_model = torch.load(args.trained_model)
print("Load weight model with {} epoch".format(trained_model["epoch"]))

model = VGG(args.model_name)
model.load_state_dict(trained_model["model_weights"])
model.to(device)
model.eval()

transform_test = transforms.Compose([
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])

def detectFace(inputImFile, outImFile):
    image = cv2.imread(inputImFile)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = haarcascade_detect.face_detect(gray_image)
    #detections = faces
    detectMsg = {}
    if detections is not None:
        for (x, y, w, h) in detections:
            roi = image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            roi_gray = Image.fromarray(np.uint8(roi_gray))
            inputs = transform_test(roi_gray)
            
            ncrops, c, ht, wt = np.shape(inputs)
            inputs = inputs.view(-1, c, ht, wt)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(ncrops, -1).mean(0)
            _, predicted = torch.max(outputs, 0)
            expression = classes[int(predicted.cpu().numpy())]
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            text = "{}".format(expression)
            
            emote.emotion = "{}".format(expression)
            
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
        cv2.imwrite(outImFile, image)
        
        detectMsg = "You are currently {0}. To continue /chat".format(expression)
    #message
#    detectMsg = {}
#    if len(detections)>0:
#        detectMsg = "You are currently {0}".format(expression)
    else:
        print("No face detected")
        print("saving into {}".format(outImFile))
        cv2.imwrite(outImFile, image)
    
        detectMsg = "No faces found please try again"
        
    return (detectFace, detectMsg)

#Code for bot
def botWorker(counter, lastConsumedUpdate):
    bot = telegram.Bot(token=botToken)
    # print bot.getMe()

    updates = bot.getUpdates(offset=lastConsumedUpdate+1, timeout=longPoolingTimeoutSec)
   
    numOfUpdates = len(updates)
    numOfNewUpdates = 0 if (numOfUpdates < 1) else updates[-1].update_id-lastConsumedUpdate #assumption of continuous counter of events

    if(numOfNewUpdates == 0):
        print('{}. No new updates'.format(counter))
        return lastConsumedUpdate
    else:
        print ('{}. There are {} new updates'.format(counter, numOfNewUpdates))

    for u in updates:
        updateId = u.update_id
        if(updateId <= lastConsumedUpdate):
            break
        print ('updateId={}, date={}'.format(u.update_id, u.message.date))


    for u in updates:
        updateId = u.update_id

        if(updateId <= lastConsumedUpdate):
            continue

        print ('updateId={}, date={}'.format(u.update_id, u.message.date))

        if u.message.photo:
            u.message.reply_text("Please wait we're processing the image...")
            print('There are {} photos in this update'.format(len(u.message.photo)))
            biggestPhoto = u.message.photo[-1]
            biggestPhotoFileId =  biggestPhoto.file_id
            print ('biggestPhoto= {}x{}, fileId={}'.format(biggestPhoto.height, biggestPhoto.width,  biggestPhoto.file_id))

            newFile = bot.getFile(biggestPhotoFileId)
            newFileUrl = newFile.file_path
            print(newFile)
            newFilePath = fromBotDir + '/' + str(updateId) + '.jpg' #build from updateId
            print('newFilePath={}'.format(newFilePath))
            # bot.download(url, newFilePath)
            # newFile.download(newFilePath)

            photoFile = urllib2.urlopen(newFileUrl)
            output = open(newFilePath,'wb')
            output.write(photoFile.read())
            output.close()

            outImFile = toBotDir + '/' + str(updateId) + '_detected' + '.jpg'
            numFaces, detectMsg = detectFace(newFilePath, outImFile)

            # To reply messages you'll always need the chat_id
            chat_id = u.message.chat_id
            print('chat_id={}'.format(chat_id))
            print('outImFile={}'.format(outImFile))
            # bot.sendPhoto(chat_id, outImFile, caption='X faces detected')
            message = bot.sendPhoto(photo=open(outImFile, 'rb'), caption=detectMsg, chat_id=chat_id)
            
        #if (u.message.text == "/respond"):
            if (emote.emotion == 'Angry'):
                chat_id = u.message.chat_id
                _1 = random.randint(1, 4)
                time.sleep(2)
                imquote = 'quotes/angry/'+ str(_1) +'.jpg'
                message = bot.sendMessage(chat_id=chat_id, text="Are you mad with me? im sorry")
                bot.sendPhoto(photo=open(imquote, 'rb'), caption='Here you go', chat_id=chat_id)
                
                
            if (emote.emotion == 'Disgust'):
                chat_id = u.message.chat_id
                time.sleep(2)
                message = bot.sendMessage(chat_id=chat_id, text="Am I smelly? Im gonna going to shower")
                
                
            if (emote.emotion == 'Fear'):
                chat_id = u.message.chat_id
                time.sleep(2)
                message = bot.sendMessage(chat_id=chat_id, text="Did you saw a ghost in me?")
                
            if (emote.emotion == 'Happy'):
                chat_id = u.message.chat_id
                time.sleep(2)
                message = bot.sendMessage(chat_id=chat_id, text="You sure have a nice day today")
                _1 = random.randint(1, 4)
                imquote = 'quotes/happy/'+ str(_1) +'.jpg'
                bot.sendPhoto(photo=open(imquote, 'rb'), caption='Here you go', chat_id=chat_id)
                
             
            if (emote.emotion == 'Sad'):
                chat_id = u.message.chat_id
                time.sleep(2)
                message = bot.sendMessage(chat_id=chat_id, text="Dont worry im always here with you")
                _1 = random.randint(1, 5)
                imquote = 'quotes/sad/'+ str(_1) +'.jpg'
                bot.sendPhoto(photo=open(imquote, 'rb'), caption='Here you go', chat_id=chat_id)
                
            if (emote.emotion == 'Suprise'):
                chat_id = u.message.chat_id
                time.sleep(2)
                message = bot.sendMessage(chat_id=chat_id, text="Whoa whoa there! did I suprise you?")
                
            if (emote.emotion == 'Neutral'):
                chat_id = u.message.chat_id
                time.sleep(2)
                message = bot.sendMessage(chat_id=chat_id, text="As calm as water")
                _1 = random.randint(1, 3)
                imquote = 'quotes/neutral/'+ str(_1) +'.jpg'
                bot.sendPhoto(photo=open(imquote, 'rb'), caption='Here you go', chat_id=chat_id)
                
                
            #'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        
        if (u.message.text == "/chat"):
            chat_id = u.message.chat_id
            if (emote.emotion == 'Angry'):
                message = bot.sendMessage(chat_id=chat_id, text="Jangan marah2 chill")
                
            if (emote.emotion == 'Happy'):
                message = bot.sendMessage(chat_id=chat_id, text="Hahaha")
                
            if (emote.emotion == 'Neutral'):
                message = bot.sendMessage(chat_id=chat_id, text="Dah makan ke belum? /dah, /belum")
                
        if (u.message.text == "/dah") and (emote.emotion == 'Neutral'):
            chat_id = u.message.chat_id
            message = bot.sendMessage(chat_id=chat_id, text="Alhamdulillah kenyang")
        if (u.message.text == "/belum") and (emote.emotion == 'Neutral'):
            chat_id = u.message.chat_id
            message = bot.sendMessage(chat_id=chat_id, text="Try masak ni: https://www.youtube.com/watch?v=wrWlXQvP4WA")
                
        
        if (u.message.text == "/start"):
            chat_id = u.message.chat_id
            bot.sendMessage(chat_id=chat_id, text="Beep Beep Bop I am a Bot!..")
            bot.sendMessage(chat_id=chat_id, text="/info for info \n/help for help")
            
        if (u.message.text == "/info"):
            chat_id = u.message.chat_id
            bot.sendMessage(chat_id=chat_id, text="This is a bot develop by using python \nYou can send your image to this bot")
            
        if (u.message.text == "/help"):
            chat_id = u.message.chat_id
            bot.sendMessage(chat_id=chat_id, text="Send your image to this bot and it we will send you back with emotion written on"
                                                  "\nMake sure your face is clearly visible and not blurry or too far away") 
            
        lastConsumedUpdate = u.update_id
        
    # To reply messages you'll always need the chat_id
    # chat_id = bot.getUpdates()[-1].message.chat_id
    # bot.sendMessage(chat_id=chat_id, text="detecting ....")

    print ('Last consumed update: {}'.format(lastConsumedUpdate))
    return lastConsumedUpdate

def main(argv = None):
    print('Starting Telegram bot backend')

    lastConsumedUpdate = lastConsumedUpdateInit
    counter = 0
    while (True):
        counter = counter + 1
        lastConsumedUpdate = botWorker(counter, lastConsumedUpdate)
        sleep(sleepIntervalSec)

if __name__ == '__main__':
    sys.exit(main())
