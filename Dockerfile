# Reproducible container (no local Python needed)
#   docker build -t multiqtl .
#   docker run --rm -v "$PWD":/data multiqtl run /data/config.yaml        # command line
#   docker run --rm -p 8765:8765 -v "$PWD":/data multiqtl gui-docker      # browser interface at http://localhost:8765
FROM python:3.11-slim
WORKDIR /opt/multiqtl
COPY . .
RUN pip install --no-cache-dir -r requirements.txt pyarrow && pip install --no-cache-dir --no-deps .
RUN printf '#!/bin/sh\nexec python -c "import multiqtl.gui as g; g.serve(8765, \\"/data/MultiQTL_runs\\", False, host=\\"0.0.0.0\\")"\n' > /usr/local/bin/gui-docker && chmod +x /usr/local/bin/gui-docker
WORKDIR /data
ENTRYPOINT ["sh", "-c", "if [ \"$1\" = gui-docker ]; then exec gui-docker; else exec multiqtl \"$@\"; fi", "--"]
CMD ["--help"]
