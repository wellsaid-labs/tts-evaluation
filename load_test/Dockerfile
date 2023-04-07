FROM loadimpact/k6
USER root

WORKDIR /app
COPY . .

ENTRYPOINT [ "k6" ]
CMD [ "run", "scenarios/tts.js" ]

