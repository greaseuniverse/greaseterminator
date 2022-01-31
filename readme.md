## GreaseTerminator

GreaseTerminator is an active, non-invasive intervention tool to patch dark patterns in operating systems and app ecosystems. Consists of `text`, `mask` and `model` hooks. This implementation corresponds to our paper [Mind-proofing Your Phone: Navigating the Digital Minefield with GreaseTerminator](https://arxiv.org/abs/2112.10699).

#### Deployment
1. Run server `python server.py` (Evaluate small-scale deployment with ngrok: `ngrok http 5000`).
2. Run interventions processing `python vision.py`.
3. Install [GreaseTerminator app on Android](), connect via ADB (wireless/wired), set server address in-app.


```
Changelog:
27-02-2021
Real-time video modification (e.g. obscenity censoring)
De-metrification of Instagram, Twitter, Facebook
Test stories bar removal across several apps
Test text censoring in images

24-02-2021
Screen darkening (for scroll lock / time lock)

23-02-2021
Hate speech detection & text censoring
Fixed resizing issue with respect to status/navigation bar
Consistent one-shot detection and removal of interface interferences (e.g. stories bar)

29-01-2021
Single shot image inpainting to remove interface interferences

25-01-2021
Text-based intervention (censor band, blacklist)
Screen relay service (flask, ngrok)
Overlap service running to render interventions; app compiled to .apk
		
```

#### Screenshots {Before, After}

##### Start page
<img src="docs/app_display.png?raw=true" height="400px"></img>

##### Content filtering (text, images)

###### Text

<img src="docs/hate_speech_before.png?raw=true" height="400px"></img>
<img src="docs/hate_speech_patched.png?raw=true" height="400px"></img>

###### Images

<img src="docs/text_in_image_censor.png?raw=true" height="400px"></img>
<img src="docs/text_in_image_censor_after.png?raw=true" height="400px"></img>

<img src="docs/image_censor.png?raw=true" height="400px"></img>
<img src="docs/image_censor_after.png?raw=true" height="400px"></img>

###### Video

<img src="docs/youtube_after.png?raw=true" height="400px"></img>

##### Interface re-interference

###### <i>Stories</i> Bar

<img src="docs/twitter_stories_before.png?raw=true" height="400px"></img>
<img src="docs/twitter_stories_patched.png?raw=true" height="400px"></img>

<img src="docs/LinkedIn_stories.jpeg?raw=true" height="400px"></img>
<img src="docs/LinkedIn_stories_after.png?raw=true" height="400px"></img>

###### De-metrification

<img src="docs/facebook_demetrication.png?raw=true" height="400px"></img>
<img src="docs/instagram_demetrication.png?raw=true" height="400px"></img>

###### Screen darkening
<img src="docs/darkening_0.png?raw=true" height="300px"></img>
<img src="docs/darkening_1.png?raw=true" height="300px"></img>
<img src="docs/darkening_2.png?raw=true" height="300px"></img>
<img src="docs/darkening_3.png?raw=true" height="300px"></img>
<img src="docs/darkening_4.png?raw=true" height="300px"></img>



