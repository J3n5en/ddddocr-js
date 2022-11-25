# ddddocr

带带弟弟OCR通用验证码识别SDK nodejs版
[原版ddddocr](https://github.com/sml2h3/ddddocr)

feature:
- [x] OCR识别
- [x] 自定义模型
- [ ] 类型识别
- [ ] 滑块

```javascript
import Ddddocr from "ddddocr";

const img = "/xxxx=="
Ddddocr.create().then(async ddddocr => {
  const result = await ddddocr.classification(Buffer.from(img, "base64"));
  console.log(result)
})
```

自定义模型
```javascript
import Ddddocr from "ddddocr";

const img = "/xxxx=="
Ddddocr.create({charsetsPath:"/xxx",onnxPath:"xxx"}).then(async ddddocr => {
  const result = await ddddocr.classification(Buffer.from(img, "base64"));
  console.log(result)
})
```
