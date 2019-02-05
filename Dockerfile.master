FROM node:lts-alpine

COPY package.json package.json
COPY package-lock.json package-lock.json

RUN npm install --prod

COPY src/ src/
COPY third_party/ third_party/

LABEL maintainer="michael@wellsaidlabs.com"

# Learn more about NODE_ENV:
# https://expressjs.com/en/advanced/best-practice-performance.html
ENV NODE_ENV production

EXPOSE 8000
CMD ["node", "src/service/master.js"]
