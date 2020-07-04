import React from 'react';
//import Slider from './Slider';
import 'react-bootstrap-range-slider/dist/react-bootstrap-range-slider.css';
import RangeSlider from 'react-bootstrap-range-slider';

class Form extends React.Component {
    state = {
        firstName: "",
        lastName: "",
        username: "",
        email: "",
        password: ""
    }

    change = e => {
        this.props.onChange({ [e.target.name]: e.target.value });
        this.setState({
            [e.target.name]: e.target.value
        })
    };

    onSubmit = e => {
        e.preventDefault();
        //this.props.onSubmit(this.state);
        this.setState({
            firstName: "",
            lastName: "",
            username: "",
            email: "",
            password: ""
        });
        this.props.onChange({
            firstName: "",
            lastName: "",
            username: "",
            email: "",
            password: ""
        });
    }

    render() {
        return (
            <form>
                <input
                    name = "firstName"
                    type = "text"
                    placeholder = "First Name"
                    value = {this.state.firstName}
                    onChange = {e => this.change(e)}
                />
                <br />
                <input
                    name = "lastName"
                    placeholder = "Last Name"
                    value = {this.state.lastName}
                    onChange = {e => this.change(e)}
                />
                <br />
                <input
                    name = "username"
                    placeholder = "Username"
                    value = {this.state.username}
                    onChange = {e => this.change(e)}
                />
                <br />
                <input
                    name = "email"
                    placeholder = "Email"
                    value = {this.state.email}
                    onChange = {e => this.change(e)}
                />
                <br />
                <input
                    name = "password"
                    type = "password"
                    placeholder = "Password"
                    value = {this.state.password}
                    onChange = {e => this.change(e)}
                />
                <br />
                <textarea 
                    name = "reviews"
                    placeholder = "Type your reviews here"
                    rows = "5" 
                    cols = " 50"
                />
                <br />
                {/* <Slider ></Slider> */}
                <button onClick={e => this.onSubmit(e)}>Submit</button>
            </form>
        );
    }
}

export default Form;