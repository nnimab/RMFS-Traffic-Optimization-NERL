extensions [ array py csv ]
breed [objects object]

objects-own [
 velocity
 acceleration
 _id
]

globals [
  _DIMENSION
  order_count
  total_energy
  average_total_energy
  robotCount
  _stop
  total_turning
  current_controller
  current_tick
  python_executable_path
]

to setup
  ca
  let result ""
  set python_executable_path "" ; 先初始化變數

  ; --- 開始讀取設定檔 ---
  if not file-exists? "python_config.txt" [
    user-message (word "錯誤：找不到 python_config.txt！請建立此檔案並填入 rmfs 環境的 Python 路徑。")
    stop
  ]

  ; 檔案存在，開啟讀取
  file-open "python_config.txt"
  if file-at-end? [
    user-message (word "錯誤：python_config.txt 為空！請填入 rmfs 環境的 Python 路徑。")
    file-close
    stop
  ]

  ; 讀取第一行
  set python_executable_path file-read-line
  file-close ; 讀完就關閉

  ; 檢查讀取的內容是否為空
  if python_executable_path = "" [
    user-message (word "錯誤：python_config.txt 讀取到的路徑為空！請檢查檔案內容。")
    stop
  ]
  ; --- 結束讀取設定檔 ---

  show (word "讀取到的 Python 路徑: " python_executable_path) ; 添加一個顯示，確認讀取成功

  ; 使用讀取到的路徑進行設定
  py:setup python_executable_path
  (py:run
    "import sys; print(f'NetLogo is using Python from: {sys.executable}')"
    "import netlogo"
    "item = netlogo.setup()")
  set result py:runresult "item"
  ask patches [
    set pcolor 8
  ]

  ;; 獲取路口信息
  (py:run
    "import netlogo"
    "intersections = netlogo.get_all_intersections()")
  let intersections py:runresult "intersections"

  ;; 標記所有路口 - 根據路口的實際ID標記

  ;; 首先標記主要路口 - 使用粉紅色突出顯示
  ask patch 15 15 [
    set pcolor 135  ;; 設置粉紅色
    ask neighbors [set pcolor 135]  ;; 為了更明顯，也將相鄰的patch設置為相同顏色
    let main-id 0
    foreach intersections [ coords ->
      let x item 0 coords
      let y item 1 coords
      let id item 2 coords
      if (x = 15 and y = 15) [
        set main-id id
      ]
    ]
    set plabel (word main-id)  ;; 添加路口實際ID
    set plabel-color white  ;; 設置標籤顏色為白色，增加可讀性
  ]

  ;; 標記其他路口 - 使用藍綠色，顯示實際ID
  foreach intersections [ coords ->
    let x item 0 coords
    let y item 1 coords
    let id item 2 coords

    ;; 跳過主要路口(15,15)，因為已經標記過了
    if not (x = 15 and y = 15) [
      ask patch x y [
        set pcolor 85  ;; 設置藍綠色
        ask neighbors [set pcolor 85]  ;; 為了更明顯，也將相鄰的patch設置為相同顏色
        set plabel (word id)  ;; 添加路口實際ID
        set plabel-color white  ;; 設置標籤顏色為白色，增加可讀性
      ]
    ]
  ]

  foreach result [
    [ x ] ->
    let id item 1 (item 0 x)
    let _heading item 1 (item 1 x)
    let _shape item 1 (item 2 x)
    let pos_x item 1 (item 5 x)
    let pos_y item 1 (item 6 x)
    let _color item 1 (item 7 x)
    create-objects 1
    [
      set _id id
      set shape _shape
      set color red
      set xcor pos_x
      set ycor pos_y
      set heading _heading
    ]
    if _shape = "turtle-2" [set robotCount robotCount + 1]
  ]
  show count objects
  set current_controller "none"
end

to go
 (py:run
    "import netlogo"
    "item = netlogo.tick()")
  let result py:runresult "item"
  foreach item 0 result [
    [ x ] ->
    let id item 1 (item 0 x)
    let _heading item 1 (item 1 x)
    let pos_x item 1 (item 5 x)
    let pos_y item 1 (item 6 x)
    let h 0
    let v 0
    let _shape item 1 (item 2 x)
    let _color item 1 (item 7 x)
    ask objects with [_id = id] [
     setxy pos_x pos_y
      set heading _heading
      set shape _shape
      set color _color
    ]
  ]
  set order_count item 2 result
  set total_energy item 1 result
  set _stop item 3 result
  set total_turning item 4 result
  set current_tick item 5 result
  set average_total_energy total_energy / robotCount
end

to setup-py
  (py:run
    "import netlogo"
    "item = netlogo.setup_py()")
end

to set-time-based
  (py:run
    "import netlogo"
    (word "result = netlogo.set_time_based_controller(" horizontal-time "," vertical-time ")"))
  let result py:runresult "result"
  ifelse result [
    set current_controller "時間基"
    show "Time-based controller set successfully"
  ] [
    show "Failed to set time-based controller"
  ]
end

to set-queue-based
  (py:run
    "import netlogo"
    (word "result = netlogo.set_queue_based_controller(" min-green-time "," bias-factor ")"))
  let result py:runresult "result"
  ifelse result [
    set current_controller "隊列基"
    show "Queue-based controller set successfully"
  ] [
    show "Failed to set queue-based controller"
  ]
end

to set-dqn
  (py:run
    "import netlogo"
    (word "result = netlogo.set_dqn_controller(" exploration-rate ")"))
  let result py:runresult "result"
  ifelse result [
    set current_controller "DQN"
    show "DQN controller set successfully"
  ] [
    show "Failed to set DQN controller"
  ]
end

to set-dqn-with-model
  (py:run
    "import netlogo"
    (word "result = netlogo.set_dqn_controller(" exploration-rate ", " model-tick ")"))
  let result py:runresult "result"
  ifelse result [
    set current_controller "DQN(loaded)"
    show (word "DQN controller set with model loaded (tick " model-tick ")")
  ] [
    show "Failed to set DQN controller or load model"
  ]
end

to set-dqn-training-mode [is-training]
  (py:run
    "import netlogo"
    (word "result = netlogo.set_dqn_training_mode(" (ifelse-value is-training ["True"] ["False"]) ")"))
  let result py:runresult "result"
  ifelse result [
    ifelse is-training [
      show "DQN controller set to training mode"
    ] [
      show "DQN controller set to evaluation mode"
    ]
  ] [
    show "Failed to set DQN controller mode, please make sure DQN controller is set"
  ]
end

to list-models
  (py:run
    "import netlogo"
    "models = netlogo.list_available_models()")
  let models py:runresult "models"
  show "Available model list:"
  foreach models [
    model -> show model
  ]
end

to list-models-by-type [controller-type]
  (py:run
    "import netlogo"
    (word "models = netlogo.list_available_models(\"" controller-type "\")"))
  let models py:runresult "models"
  show (word "Available " controller-type " model list:")
  foreach models [
    model -> show model
  ]
end

to set-nerl-controller
  (py:run
    "import netlogo"
    (word "result = netlogo.set_nerl_controller(" exploration-rate ")"))
  let result py:runresult "result"
  ifelse result [
    set current_controller "NERL"
    show "NERL controller set successfully"
  ] [
    show "Failed to set NERL controller"
  ]
end

to set-nerl-with-model
  (py:run
    "import netlogo"
    (word "result = netlogo.set_nerl_controller(" exploration-rate ", " model-tick ")"))
  let result py:runresult "result"
  ifelse result [
    set current_controller "NERL(loaded)"
    show (word "NERL controller set with model loaded (tick " model-tick ")")
  ] [
    show "Failed to set NERL controller or load model"
  ]
end

to set-nerl-training-mode [is-training]
  (py:run
    "import netlogo"
    (word "result = netlogo.set_nerl_training_mode(" (ifelse-value is-training ["True"] ["False"]) ")"))
  let result py:runresult "result"
  ifelse result [
    ifelse is-training [
      show "NERL controller set to training mode"
    ] [
      show "NERL controller set to evaluation mode"
    ]
  ] [
    show "Failed to set NERL controller mode, please make sure NERL controller is set"
  ]
end

to generate-report
  (py:run
    "import netlogo"
    "result = netlogo.generate_report()")
  let result py:runresult "result"
  ifelse result [
    show "Performance report generated successfully!"
  ] [
    show "Failed to generate performance report, please check console log."
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
15
10
758
484
-1
-1
15.0
1
10
1
1
1
0
1
1
1
0
48
0
30
0
0
1
ticks
15.0

BUTTON
775
100
858
133
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
910
325
974
359
go
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
775
325
872
358
go-forever
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
1260
10
1423
67
Order
order_count
17
1
14

MONITOR
1260
96
1422
153
Total Energy
total_energy
17
1
14

MONITOR
1260
173
1425
230
Average Energy
average_total_energy
17
1
14

MONITOR
1260
248
1425
305
Stop and Go
_stop
17
1
14

MONITOR
1260
323
1425
380
Total Turning
total_turning
17
1
14

BUTTON
1115
360
1197
393
Setup Py
setup-py
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
775
145
855
179
時間基控制器
set-time-based
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
775
185
855
219
隊列基控制器
set-queue-based
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
775
230
855
263
DQN控制器
set-dqn
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
775
274
855
307
NERL控制器
set-nerl-controller
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
775
10
940
67
當前控制器
current_controller
17
1
14

SLIDER
880
105
1052
138
horizontal-time
horizontal-time
10
100
30.0
5
1
NIL
HORIZONTAL

SLIDER
1061
105
1233
138
vertical-time
vertical-time
10
100
20.0
5
1
NIL
HORIZONTAL

SLIDER
880
148
1052
181
min-green-time
min-green-time
1
30
1.0
1
1
NIL
HORIZONTAL

SLIDER
1061
148
1233
181
bias-factor
bias-factor
1
3
1.5
0.1
1
NIL
HORIZONTAL

SLIDER
880
191
1052
224
exploration-rate
exploration-rate
0
1
0.6
0.05
1
NIL
HORIZONTAL

SLIDER
1060
190
1235
223
model-tick
model-tick
0
50000
5000.0
5000
1
NIL
HORIZONTAL

BUTTON
1115
320
1195
353
查看模型
list-models
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
1061
234
1194
267
DQN(加載模型)
set-dqn-with-model
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
1061
275
1194
308
NERL(加載模型)
set-nerl-with-model
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
880
275
994
308
NERL訓練模式開
set-nerl-training-mode true
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
994
275
1051
308
關
set-nerl-training-mode false
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
880
235
994
268
DQN訓練模式開
set-dqn-training-mode true
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
995
235
1052
268
關
set-dqn-training-mode false
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
1085
10
1220
67
Current Tick
current_tick
4
1
14

BUTTON
775
375
974
408
生成性能報告
generate-report
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

arrow-down
true
0
Line -16777216 false 75 150 150 75
Line -16777216 false 225 150 150 75

arrow-left
true
0
Line -16777216 false 150 225 75 150
Line -16777216 false 75 150 150 75

arrow-right
true
0
Line -16777216 false 150 75 225 150
Line -16777216 false 225 150 150 225

arrow-up
true
0
Line -16777216 false 75 150 150 225
Line -16777216 false 150 225 225 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

empty-space
true
0

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

full square
true
0
Rectangle -7500403 false true 0 0 300 300
Rectangle -13791810 true false 0 0 420 360
Rectangle -13345367 false false 0 0 300 300

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

intersection
true
0
Rectangle -2674135 true false 0 0 390 360

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person-red
false
0
Circle -8630108 false false 120 60 60
Line -8630108 false 150 120 150 225
Line -8630108 false 150 135 105 180
Line -8630108 false 150 135 195 180
Line -8630108 false 150 225 195 285
Line -8630108 false 150 225 105 285

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

rail
true
0
Rectangle -16777216 true false 0 135 315 165

rail-corner
true
0
Rectangle -16777216 true false 135 135 315 165
Rectangle -16777216 true false 135 135 165 330

rail-triangle
true
0
Rectangle -16777216 true false 0 135 315 165
Rectangle -16777216 true false 135 135 165 330

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -2674135 true false 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

turtle-2
true
0
Rectangle -7500403 true true 70 45 227 120
Polygon -7500403 true true 150 8 118 10 96 17 90 30 75 135 75 195 90 210 150 210 210 210 225 195 225 135 209 30 201 17 179 10
Polygon -16777216 true false 94 135 118 119 184 119 204 134 193 141 110 141
Line -16777216 false 130 14 168 14
Line -16777216 false 130 18 168 18
Line -16777216 false 130 11 168 11
Line -16777216 false 185 29 194 112
Line -16777216 false 115 29 106 112
Line -16777216 false 195 225 210 240
Line -16777216 false 105 225 90 240
Polygon -16777216 true false 210 195 195 195 195 150 210 143
Polygon -16777216 false false 90 143 90 195 105 195 105 150 90 143
Polygon -16777216 true false 90 195 105 195 105 150 90 143
Line -7500403 true 210 180 195 180
Line -7500403 true 90 180 105 180
Line -16777216 false 212 44 213 124
Line -16777216 false 88 44 87 124
Line -16777216 false 223 130 193 112
Rectangle -7500403 true true 225 133 244 139
Rectangle -7500403 true true 56 133 75 139
Rectangle -7500403 true true 120 210 180 240
Rectangle -7500403 true true 93 238 210 270
Rectangle -16777216 true false 200 217 224 278
Rectangle -16777216 true false 76 217 100 278
Circle -16777216 false false 135 240 30
Line -16777216 false 77 130 107 112
Rectangle -16777216 false false 107 149 192 210
Rectangle -1 true false 180 9 203 17
Rectangle -1 true false 97 9 120 17

turtle-3
true
0
Rectangle -7500403 true true 70 45 227 120
Polygon -7500403 true true 150 8 118 10 96 17 90 30 75 135 75 195 90 210 150 210 210 210 225 195 225 135 209 30 201 17 179 10
Polygon -16777216 true false 94 135 118 119 184 119 204 134 193 141 110 141
Line -16777216 false 130 14 168 14
Line -16777216 false 130 18 168 18
Line -16777216 false 130 11 168 11
Line -16777216 false 185 29 194 112
Line -16777216 false 115 29 106 112
Line -16777216 false 195 225 210 240
Line -16777216 false 105 225 90 240
Polygon -16777216 true false 210 195 195 195 195 150 210 143
Polygon -16777216 false false 90 143 90 195 105 195 105 150 90 143
Polygon -16777216 true false 90 195 105 195 105 150 90 143
Line -7500403 true 210 180 195 180
Line -7500403 true 90 180 105 180
Line -16777216 false 212 44 213 124
Line -16777216 false 88 44 87 124
Line -16777216 false 223 130 193 112
Rectangle -7500403 true true 225 133 244 139
Rectangle -7500403 true true 56 133 75 139
Rectangle -7500403 true true 120 210 180 240
Rectangle -7500403 true true 93 238 210 270
Rectangle -16777216 true false 200 217 224 278
Rectangle -16777216 true false 76 217 100 278
Circle -16777216 false false 135 240 30
Line -16777216 false 77 130 107 112
Rectangle -16777216 false false 107 149 192 210
Rectangle -1 true false 180 9 203 17
Rectangle -1 true false 97 9 120 17

turtle-occupied
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -2674135 true false 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

turtle-tp
true
0
Rectangle -7500403 true true 70 45 227 120
Polygon -7500403 true true 150 8 118 10 96 17 90 30 75 135 75 195 90 210 150 210 210 210 225 195 225 135 209 30 201 17 179 10
Polygon -16777216 true false 94 135 118 119 184 119 204 134 193 141 110 141
Line -16777216 false 130 14 168 14
Line -16777216 false 130 18 168 18
Line -16777216 false 130 11 168 11
Line -16777216 false 185 29 194 112
Line -16777216 false 115 29 106 112
Line -16777216 false 195 225 210 240
Line -16777216 false 105 225 90 240
Polygon -16777216 true false 210 195 195 195 195 150 210 143
Polygon -16777216 false false 90 143 90 195 105 195 105 150 90 143
Polygon -16777216 true false 90 195 105 195 105 150 90 143
Line -7500403 true 210 180 195 180
Line -7500403 true 90 180 105 180
Line -16777216 false 212 44 213 124
Line -16777216 false 88 44 87 124
Line -16777216 false 223 130 193 112
Rectangle -7500403 true true 225 133 244 139
Rectangle -7500403 true true 56 133 75 139
Rectangle -7500403 true true 120 210 180 240
Rectangle -7500403 true true 93 238 210 270
Rectangle -16777216 true false 200 217 224 278
Rectangle -16777216 true false 76 217 100 278
Circle -16777216 false false 135 240 30
Line -16777216 false 77 130 107 112
Rectangle -16777216 false false 107 149 192 210
Rectangle -1 true false 180 9 203 17
Rectangle -1 true false 97 9 120 17
Polygon -13345367 true false 105 105 120 30 105 105
Polygon -13345367 true false 120 105 135 30 165 30 180 105 120 105
Polygon -13345367 true false 90 120 105 105 105 30 90 120
Polygon -13345367 true false 210 120 195 105 195 30 210 120

wall
true
0
Rectangle -16777216 true false 0 0 345 300

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
