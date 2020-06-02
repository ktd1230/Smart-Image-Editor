<template>
  <div id="app">
    <canvas id="canvas" width="800" height="600"></canvas>
    <input type="text" v-model="exampleContent" />
    <span>{{ exampleContent }}</span>
</div>
</template>

<script>
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

ctx.fillStyle = "black";
ctx.font="20px Georgia";
ctx.fillText("Hello World!",10,50);
export default {
    el: '#app',
  data: {
    'exampleContent': 'This is TEXT'
  },
  watch: {
    exampleContent: function(val, oldVal) {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.fillStyle = "black";
      ctx.font="20px Georgia";
      ctx.fillText(this.exampleContent,10,50);
    }
  }

}
</script>

<style>

</style>