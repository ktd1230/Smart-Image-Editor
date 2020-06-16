<template>
  <v-app>
    <v-layout column="column" wrap="wrap" class="my-12" align-center="align-center" style="flex:0;" data-html2canvas-ignore="true">
        <v-flex xs12="xs12" sm4="sm4" class="mt-4">
            <div class="text-center">
                <h2 class="font2">내서재 도서 목록</h2>
                <br>
                <span class="subheading font1">내가 만든 도서를 확인할 수 있습니다.</span><br/>
                <span class="subheading font1">하나뿐인 기록을 내서재에서 간직해보세요.</span>
            </div>
        </v-flex>
    </v-layout>
    <div class="bg row d-flex justify-center">
        <br>
        <div v-for="(story, i) in storys" :key="i" class="col-md-5 col-lg-3" style="margin:20px;">
            <div icon color="#C62828" style="outline:0" class="text-center">
                <v-icon style="color:red">mdi-heart</v-icon>
                <span style="color:red">({{story.like_users.length}})</span>
            </div>
            <v-hover>
                <template v-slot:default="{ hover }" style="position:absolute;z-index:0;">
                <div class="card text-white" style="background-color:unset;">
                    <img class="card-img" height="230" :src="`http://13.124.246.175:8000/media/${first_image[i]}`" alt="" style="width: auto;object-fit:cover;">
                    <div class="card-img-overlay text-center px-0">
                        <h3 v-if="story.title" class="mt-4 px-2 black card-title" style="margin-left:90px; margin-right:90px;">{{story.title}}</h3>
                        <h3 v-else class="mt-4 card-title">...</h3>
                        <p class="py-1 subheading black card-text" style="margin-top:100px;">{{story.create_at}}</p>
                        <v-fade-transition>
                            <v-overlay
                                v-if="hover"
                                absolute
                                color="#036358"
                            >
                                <v-btn @click="detail(story)">자세히 보기</v-btn>
                            </v-overlay>
                        </v-fade-transition>
                    </div>
                </div>
                </template>
            </v-hover>
        </div>
      </div>
  </v-app>
</template>

<script>
import axios_common from '../../axios_common';
import { mapGetters } from 'vuex';

export default {
    computed: {
        ...mapGetters([
        'requestHeader',
        ])
    },
    data() {
        return {
            storys : [],
            first_image : [],
            books: [
                {
                    title: "책1",
                    img: 'https://i.pinimg.com/564x/7b/8f/1e/7b8f1e3287a38700d88d1596c1144c9b.jpg',
                    date: '2020-04-20'
                },
                {
                    title: '책2',
                    img: 'https://elib.seoul.go.kr/resources/images/YES24/Msize/8638926M.jpg',
                    date: '2020-04-22'
                },
                {
                    title: '책3',
                    img: 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTOfEu4PQtrmmW5gkjA0RBT-8rN3eSBKVZnLdZpVxTveJozU7pm&usqp=CAU',
                    date: '2020-04-22'
                },
                {
                    title: "책1",
                    img: 'https://i.pinimg.com/564x/7b/8f/1e/7b8f1e3287a38700d88d1596c1144c9b.jpg',
                    date: '2020-04-20'
                },
                {
                    title: '책2',
                    img: 'https://elib.seoul.go.kr/resources/images/YES24/Msize/8638926M.jpg',
                    date: '2020-04-22'
                },
                {
                    title: '책3',
                    img: 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTOfEu4PQtrmmW5gkjA0RBT-8rN3eSBKVZnLdZpVxTveJozU7pm&usqp=CAU',
                    date: '2020-04-22'
                }
            ],
        }
    },   
    methods: {
        detail(story) {
            this.$router.push({ name: 'detail', params: {story: story}});
        }
    },
    mounted() {
        axios_common.get('/sub3/mystory/', this.requestHeader)
            .then(response => {
                this.first_image = response.data.first_image
                for (var i=0; i < response.data.data.length; i++){
                    this.storys.push(response.data.data[i])
                }
            })
            .catch(error => console.log(error))
    },
}
</script>
<style scoped>
.bg {
    padding: 30px;
    background: rgb(244, 242, 243);
}
#img {
  transition: transform 0.2s; /* Animation */
}
#img:hover {
  opacity: 50%;
  -webkit-transform: scale(1.2);
  transform: scale(1.2);
}
.font1 {font-family: 'Do Hyeon', sans-serif;}
.font2 {font-family: 'Black Han Sans', sans-serif;}
</style>
