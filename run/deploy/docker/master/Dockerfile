FROM node:lts-alpine

COPY package.json package.json
COPY package-lock.json package-lock.json

RUN npm install --prod

COPY run/ run/

LABEL maintainer="michael@wellsaidlabs.com"

# Learn more about NODE_ENV:
# https://expressjs.com/en/advanced/best-practice-performance.html
# TODO (michael): Apply these security best practices:
# https://snyk.io/blog/10-docker-image-security-best-practices/
ENV NODE_ENV production

EXPOSE 8000
CMD ["node", "--trace-warnings", "run/deploy/master.js"]
