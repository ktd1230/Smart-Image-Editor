<template>
  <div>
    <v-container class="grey lighten-5 pa-10">
      <v-row>
        <v-col md="6" xs="12" sm="12" class="pa-0">
          <v-img :src="`${back_server}:8000/media/${original_image}`" height="600"></v-img>
          <!-- <v-carousel
          height="600"
          >
            <v-carousel-item
              v-for="(image,i) in display_images"
              :key="i"
              :src="`${back_server}:8000/media/${image}`"
              reverse-transition="fade-transition"
              transition="fade-transition"
            ></v-carousel-item>
          </v-carousel>-->
        </v-col>
        <v-col md="6" xs="12" sm="12">
          <v-container
            fluid
            pa-0
            id="scroll-target"
            style="max-height: 580px"
            class="overflow-y-auto overflow-x-hidden pl-3"
          >
            <v-row>
                <v-col cols="12" v-for="(item,i) in masked_images" :key="i">
                    <v-hover v-slot:default="{ hover }">
                        <v-card class="sampletext" max-width = 95%>
                            <v-card-title>{{i+1}}번 객체</v-card-title>
                            <v-img
                                :src="`${back_server}:8000/media/${item}`"
                                height="200px"
                            >                                
                            </v-img>
                                <!-- <v-fade-transition> -->
                                    <v-overlay v-if="!selected[i] && hover" absolute="absolute" color="#036358">
                                        <v-btn @click="select(i)">선택하기</v-btn>
                                        <v-btn @click="save_mask(i)">저장하기</v-btn>
                                    </v-overlay>
                                    <v-overlay v-if="selected[i]" absolute="absolute" color="#036358">
                                        <!-- <v-btn @click="select(i)">선택하기</v-btn> -->
                                        
                                    </v-overlay>
                                <!-- </v-fade-transition> -->
                            
                        </v-card>
                    </v-hover>
                  
                </v-col>
                <!-- <v-col cols="12" v-if="selectedImage >= 0">
                    <v-card class="sampletext" max-width = 95%
                    data-aos="fade-down"
                    data-aos-duration="3000"
                    data-aos-delay="150"
                    >
                        <v-card-title>
                            <v-col cols="5">
                                <v-text-field 
                                single-line
                                :disabled="!enabled"
                                v-model="title"
                                >
                                <template v-slot:label >
                                <strong>{{selectedImage+1}}번</strong> 샘플
                                </template>    
                                </v-text-field>
                            </v-col>
                        </v-card-title>
                        <v-card-text class="text--primary">
                        <v-textarea
                        filled
                        auto-grow
                        rows="2"
                        row-height="20"
                        :value="items.text[selectedImage]"
                        :disabled="!enabled"
                        v-model="content"
                        ></v-textarea>
                        <v-col align="end">
                        <v-btn @click="back">이전으로</v-btn>
                        <v-btn color="#ff5989" dark @click="enabled=!enabled">수정하기</v-btn>
                        <v-btn color="#4CAF50" dark @click="save">저장하기</v-btn>
                        </v-col>
                        </v-card-text>
                    </v-card>
              </v-col>-->
            </v-row>
          </v-container>
        </v-col>
      </v-row>
    </v-container>
    <v-container>
      <div class="text-center">
        <v-btn class="ma-2" tile color="indigo" dark @click="mask_rcnn">테두리 얻어내기</v-btn>
        <v-btn class="ma-2" tile color="indigo" dark @click="resolution_up_edsr">해상도 올리기(edsr)</v-btn>
        <v-btn class="ma-2" tile color="indigo" dark @click="resolution_up_prsr">해상도 올리기(prosr)</v-btn>
        <v-btn class="ma-2" tile color="indigo" dark @click="inpainting">객체 삭제</v-btn>
        <v-dialog v-model="dialog" max-width="450">
          <v-card>
            <v-card-title class="headline">결과</v-card-title>
            <v-img :src="`${back_server}:8000/media/${display_images}`" height="200px"></v-img>
            <v-spacer></v-spacer>
            <v-btn color="green darken-1" text @click="save_result">저장하기</v-btn>

            <v-btn color="green darken-1" text @click="dialog = false">취소하기</v-btn>
          </v-card>
        </v-dialog>
      </div>
    </v-container>
  </div>
</template>


<script>
//import router from "../../router"
import axios_common from "../../axios_common";
import { mapGetters } from "vuex";
import axios from "axios";
import bus from "../../utiles/bus.js";

export default {
  props: ["response"],
  computed: {
    ...mapGetters(["isAuthenticated", "requestHeader", "userId", "username"])
  },
  mounted() {
    // console.log(this.$vuetify.breakpoint)
    // bus.$emit('end:spinner');
    console.log("original_image", this.original_image);
  },
  created() {
    // console.log(this.$route.response)
    // console.log("result창")
    console.log(this.response);
    //this.back_server = "http://13.124.246.175"
    this.back_server = "http://localhost";
  },
  data() {
    return {
      selected: [],
      dialog: false,
      overlay: false,
      selectedImage: -1,
      meta: this.$route.response,
      enabled: false,
      edit_content: false,
      title: "",
      content: "",
      // this.response.text,
      original_image: this.response.image[0],
      display_images: this.response.image,
      masked_images: [],
      mask: []
    };
  },
  methods: {
    select(num) {
        this.selectedImage=num
        this.selected.splice(0, this.selected.length, false)
        this.selected[num] = true
      console.log("event.target", event.target);
      this.selectedImage = num;
      //this.display_images[0]=this.masked_images[this.selectedImage]
      console.log(
        "In select funtion display_images[0]",
        this.display_images[0]
      );
      console.log("this.selectedImage=num", this.selectedImage);
    },
    back() {
      this.selectedImage = -1;
    },
    forceFileDownload(response) {
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "file.png"); //or any other extension
      document.body.appendChild(link);
      link.click();
    },
    downloadWithVueResource(filename) {
      axios({
        method: "get",
        url: this.back_server + ":8000/media/" + filename,
        responseType: "arraybuffer"
      })
        .then(response => {
          this.forceFileDownload(response);
        })
        .catch(() => console.log("error occured"));
    },
    save_mask(number) {
      var file_name = "";
      file_name = this.masked_images[number];
      console.log("save_mask : filename", file_name);
      this.downloadWithVueResource(file_name);
    },
    save_result() {
      this.downloadWithVueResource(this.display_images);
      this.dialog = false;
    },
    mask_rcnn() {
      bus.$emit("start:loading");
      axios_common
        .post(
          "/sub3/mask_rcnn/",
          { img: this.original_image },
          this.requestHeader
        )
        .then(response => {
          console.log(response.data);
          this.masked_images = response.data.masked_images;
          this.mask = response.data.mask;
          bus.$emit("end:loading");
        })
        .catch(error => console.log(error));
    },
    resolution_up_edsr() {
      bus.$emit("start:loading");
      axios_common
        .post(
          "/sub3/resolution_up_edsr/",
          { img: this.original_image },
          this.requestHeader
        )
        .then(response => {
          console.log("resolution", response.data.resolution_up);
          this.display_images = response.data.resolution_up[0];
          //this.display_images.push(response.data.resolution_up[0])
          console.log("this.display_images[0]", this.display_images[0]);
          console.log("this.display_images", this.display_images);
          this.dialog = true;
          bus.$emit("end:loading");
        })
        .catch(error => console.log(error));
    },
    resolution_up_prsr() {
      bus.$emit("start:loading");
      axios_common
        .post(
          "/sub3/resolution_up_prosr/",
          { img: this.original_image },
          this.requestHeader
        )
        .then(response => {
          console.log("resolution", response.data.resolution_up);
          //this.display_images.push(response.data.resolution_up);
          this.display_images = response.data.resolution_up;
          console.log("this.display_images", this.display_images);
          this.dialog = true;
          bus.$emit("end:loading");
        })
        .catch(error => console.log(error));
    },
    inpainting() {
      if (this.selectedImage == -1){
          window.alert("객체를 선택해주세요")
          return
      }
      bus.$emit("start:loading");
      console.log(
        "this.mask[this.selectedImage]",
        this.mask[this.selectedImage]
      );
      axios_common
        .post(
          "/sub3/inpainting/",
          { img: this.original_image, mask: this.mask[this.selectedImage] },
          this.requestHeader
        )
        .then(response => {
          console.log("inpainting", response.data);
          this.display_images = response.data.inpainting;
          //this.display_images.push(response.data.inpainting)
          this.dialog = true;
          bus.$emit("end:loading");
        })
        .catch(error => console.log(error));
    }
  }
};
</script>
