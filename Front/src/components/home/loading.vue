<template>
    <div class="loader mx-auto" v-if="loading">
        <div class="box"></div>
        <div class="box"></div>
    </div>
</template>

<script>
export default {
    props: {
        loading: {
            type: Boolean,
            required: true,
        }
    },
}
</script>

<style>
.loader {
  display: -webkit-inline-box;
  display: inline-flex;
  flex-wrap: wrap;
  width: 90px;
  height: 90px;
  -webkit-transform-style: preserve-3d;
          transform-style: preserve-3d;
  -webkit-transform: rotateX(45deg) rotate(45deg);
          transform: rotateX(45deg) rotate(45deg);
}

.box {
  position: absolute;
  top: 0;
  left: 0;
  width: 30px;
  height: 30px;
  background: #FED750;
  box-shadow: 112.5px 112.5px 20px #000;
  -webkit-animation: move 2s ease-in-out infinite both;
          animation: move 2s ease-in-out infinite both;
  -webkit-transform-style: preserve-3d;
          transform-style: preserve-3d;
}
.box:nth-child(1) {
  -webkit-animation-delay: -1s;
          animation-delay: -1s;
}
.box:nth-child(2) {
  -webkit-animation-delay: -2s;
          animation-delay: -2s;
}
.box:nth-child(3) {
  -webkit-animation-delay: -3s;
          animation-delay: -3s;
}
.box:before, .box:after {
  display: block;
  content: "";
  position: absolute;
  width: 30px;
  height: 30px;
}
.box:before {
  top: 100%;
  left: 0;
  background: #E6A32F;
  -webkit-transform-origin: center top;
          transform-origin: center top;
  -webkit-transform: rotateX(-90deg);
          transform: rotateX(-90deg);
}
.box:after {
  top: 0;
  left: 100%;
  background: #C87932;
  -webkit-transform-origin: center left;
          transform-origin: center left;
  -webkit-transform: rotateY(90deg);
          transform: rotateY(90deg);
}

@-webkit-keyframes move {
  0%, 100% {
    -webkit-transform: none;
            transform: none;
  }
  12.5% {
    -webkit-transform: translate(30px, 0);
            transform: translate(30px, 0);
  }
  25% {
    -webkit-transform: translate(60px, 0);
            transform: translate(60px, 0);
  }
  37.5% {
    -webkit-transform: translate(60px, 30px);
            transform: translate(60px, 30px);
  }
  50% {
    -webkit-transform: translate(60px, 60px);
            transform: translate(60px, 60px);
  }
  62.5% {
    -webkit-transform: translate(30px, 60px);
            transform: translate(30px, 60px);
  }
  75% {
    -webkit-transform: translate(0, 60px);
            transform: translate(0, 60px);
  }
  87.5% {
    -webkit-transform: translate(0, 30px);
            transform: translate(0, 30px);
  }
}

@keyframes move {
  0%, 100% {
    -webkit-transform: none;
            transform: none;
  }
  12.5% {
    -webkit-transform: translate(30px, 0);
            transform: translate(30px, 0);
  }
  25% {
    -webkit-transform: translate(60px, 0);
            transform: translate(60px, 0);
  }
  37.5% {
    -webkit-transform: translate(60px, 30px);
            transform: translate(60px, 30px);
  }
  50% {
    -webkit-transform: translate(60px, 60px);
            transform: translate(60px, 60px);
  }
  62.5% {
    -webkit-transform: translate(30px, 60px);
            transform: translate(30px, 60px);
  }
  75% {
    -webkit-transform: translate(0, 60px);
            transform: translate(0, 60px);
  }
  87.5% {
    -webkit-transform: translate(0, 30px);
            transform: translate(0, 30px);
  }
}
</style>